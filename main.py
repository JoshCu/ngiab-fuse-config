#!/usr/bin/env python
"""
Virtual configuration directory using FUSE
Dynamically serves model configuration files from templates and .gpkg data
"""

import errno
import logging
import math
import os
import sqlite3
import stat
from datetime import datetime
from pathlib import Path
from typing import Optional

import fuse
from fuse import Fuse

# Optional: for coordinate transformation if pyproj is available
try:
    from pyproj import Transformer

    HAS_PYPROJ = True
except ImportError:
    HAS_PYPROJ = False

if not hasattr(fuse, "__version__"):
    raise RuntimeError("your fuse-py doesn't know of fuse.__version__, probably it's too old.")

fuse.fuse_python_api = (0, 2)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MyStat(fuse.Stat):
    def __init__(self):
        self.st_mode = 0
        self.st_ino = 0
        self.st_dev = 0
        self.st_nlink = 0
        self.st_uid = os.getuid()
        self.st_gid = os.getgid()
        self.st_size = 0
        self.st_atime = 0
        self.st_mtime = 0
        self.st_ctime = 0


class ConfigFS(Fuse):
    """Virtual filesystem for serving configuration files"""

    # Model name mappings (filesystem name -> template name)
    MODEL_MAPPINGS = {"CFE": "cfe", "lstm": "lstm", "NOAH-OWP-M": "noahowp"}

    def __init__(self, *args, **kw):
        Fuse.__init__(self, *args, **kw)

        # Configuration parameters
        self.data_dir = None
        self.config_dir = None
        self.gpkg_path = None
        self.template_dir = None
        self.start_time = None
        self.end_time = None

        # Cache for generated configs
        self.config_cache = {}
        self.divide_data = {}

        # Discovered models and templates
        self.models = {}  # model_name -> (template_content, file_extension)

    def initialize_fs(
        self, data_dir: str, start_time: str, end_time: str, template_dir: Optional[str] = None
    ):
        """Initialize the filesystem with configuration"""
        self.data_dir = Path(data_dir)
        self.config_dir = self.data_dir / "config"

        # Find the gpkg file
        gpkg_files = list(self.config_dir.glob("*.gpkg"))
        if not gpkg_files:
            raise ValueError(f"No .gpkg file found in {self.config_dir}")
        self.gpkg_path = gpkg_files[0]
        logger.info(f"Using geopackage: {self.gpkg_path}")

        # Template directory
        self.template_dir = Path(template_dir) if template_dir else Path("./templates")

        # Time settings
        self.start_time = datetime.strptime(start_time, "%Y-%m-%d %H:%M:%S")
        self.end_time = datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")

        # Discover and load templates
        self.discover_models()

        # Load divide data from gpkg
        self.load_divide_data()

        logger.info(
            f"Initialized ConfigFS with {len(self.divide_data)} catchments and {len(self.models)} models"
        )

    def discover_models(self):
        """Discover available models from template files"""
        if not self.template_dir.exists():
            logger.warning(f"Template directory not found: {self.template_dir}")
            return

        # Look for all *_template.* files
        for template_file in self.template_dir.glob("*_template.*"):
            # Extract model name and extension
            name_parts = template_file.stem.split("_template")[0]
            extension = template_file.suffix

            # Read template content
            with open(template_file, "r") as f:
                content = f.read()

            # Determine filesystem directory name
            if name_parts in self.MODEL_MAPPINGS.values():
                # Find the filesystem name for this template name
                fs_name = next(
                    (k for k, v in self.MODEL_MAPPINGS.items() if v == name_parts), name_parts
                )
            else:
                # Use the template name as-is
                fs_name = name_parts

            self.models[fs_name] = (content, extension)
            logger.info(f"Loaded model template: {fs_name} ({template_file.name})")

    def load_divide_data(self):
        """Load divide attributes from geopackage"""
        try:
            with sqlite3.connect(self.gpkg_path) as conn:
                cursor = conn.cursor()

                # Check what tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                logger.info(f"Available tables: {tables}")

                # Find the attributes table
                attr_table = None
                for possible_name in [
                    "divide-attributes",
                    "divide_attributes",
                    "divides-attributes",
                    "divides_attributes",
                ]:
                    if possible_name in tables:
                        attr_table = possible_name
                        break

                if attr_table:
                    # Get all divide attributes
                    query = f"""
                    SELECT d.divide_id, d.areasqkm, da.*
                    FROM divides AS d
                    LEFT JOIN '{attr_table}' AS da ON d.divide_id = da.divide_id
                    """

                    cursor.execute(query)
                    columns = [desc[0] for desc in cursor.description]

                    for row in cursor.fetchall():
                        divide_id = row[0]
                        self.divide_data[divide_id] = dict(zip(columns, row))

                    # Try to get coordinates
                    cursor.execute("PRAGMA table_info(divides)")
                    divide_columns = [col[1] for col in cursor.fetchall()]

                    if "centroid_x" in divide_columns and "centroid_y" in divide_columns:
                        cursor.execute("SELECT divide_id, centroid_x, centroid_y FROM divides")
                        for row in cursor.fetchall():
                            divide_id = row[0]
                            if divide_id in self.divide_data:
                                x, y = row[1], row[2]
                                # Simple transformation or use as-is
                                self.divide_data[divide_id]["longitude"] = (
                                    x if abs(x) <= 180 else -95.0
                                )
                                self.divide_data[divide_id]["latitude"] = (
                                    y if abs(y) <= 90 else 40.0
                                )
                    else:
                        # No coordinate columns, use defaults
                        for divide_id in self.divide_data:
                            self.divide_data[divide_id]["longitude"] = -95.0
                            self.divide_data[divide_id]["latitude"] = 40.0

                else:
                    # Just load basic divide info
                    cursor.execute("SELECT divide_id, areasqkm FROM divides")
                    for row in cursor.fetchall():
                        divide_id = row[0]
                        self.divide_data[divide_id] = {
                            "divide_id": divide_id,
                            "areasqkm": row[1],
                            "longitude": -95.0,
                            "latitude": 40.0,
                        }

                logger.info(f"Loaded {len(self.divide_data)} divides")

        except Exception as e:
            logger.error(f"Error loading divide data: {e}")
            raise

    def safe_get(self, row: dict, key: str, default=0):
        """Safely get a value from row dict, handling various key formats"""
        if key in row and row[key] is not None:
            return row[key]
        # Try with quotes
        quoted_key = f'"{key}"'
        if quoted_key in row and row[quoted_key] is not None:
            return row[quoted_key]
        # Try with underscores instead of dots
        underscore_key = key.replace(".", "_")
        if underscore_key in row and row[underscore_key] is not None:
            return row[underscore_key]
        return default

    def generate_config(self, model: str, divide_id: str) -> bytes:
        """Generate configuration for a specific model and catchment"""
        if model not in self.models:
            return f"Model {model} not found\n".encode("utf-8")

        row = self.divide_data.get(divide_id)
        if not row:
            return f"Divide {divide_id} not found\n".encode("utf-8")

        template, _ = self.models[model]

        # Prepare common substitutions
        substitutions = {
            "divide_id": divide_id,
            "area_sqkm": self.safe_get(row, "areasqkm", 100),
            "areasqkm": self.safe_get(row, "areasqkm", 100),
            "lat": self.safe_get(row, "latitude", 40.0),
            "latitude": self.safe_get(row, "latitude", 40.0),
            "lon": self.safe_get(row, "longitude", -95.0),
            "longitude": self.safe_get(row, "longitude", -95.0),
            "start_datetime": self.start_time.strftime("%Y%m%d%H%M"),
            "end_datetime": self.end_time.strftime("%Y%m%d%H%M"),
        }

        # Model-specific substitutions
        if model == "CFE":
            mean_zmax = self.safe_get(row, "mean.Zmax", 11)
            max_gw_storage = (mean_zmax / 1000) if mean_zmax else 0.011

            substitutions.update(
                {
                    "bexp": self.safe_get(row, "mode.bexp_soil_layers_stag=2", 6.0),
                    "dksat": self.safe_get(row, "geom_mean.dksat_soil_layers_stag=2", 5e-6),
                    "psisat": self.safe_get(row, "geom_mean.psisat_soil_layers_stag=2", 0.5),
                    "slope": self.safe_get(row, "mean.slope_1km", 0.01),
                    "smcmax": self.safe_get(row, "mean.smcmax_soil_layers_stag=2", 0.45),
                    "smcwlt": self.safe_get(row, "mean.smcwlt_soil_layers_stag=2", 0.15),
                    "max_gw_storage": f"{max_gw_storage:.6f}",
                    "gw_Coeff": self.safe_get(row, "mean.Coeff", 0.0018),
                    "gw_Expon": self.safe_get(row, "mode.Expon", 1.0),
                    "gw_storage": 0.05,
                    "refkdt": self.safe_get(row, "mean.refkdt", 3.0),
                }
            )

        elif model == "lstm":
            slope_deg = self.safe_get(row, "mean.slope", 45)
            flipped_slope = abs(slope_deg - 90)
            slope_mpkm = math.tan(math.radians(flipped_slope)) * 1000

            substitutions.update(
                {
                    "slope_mean": slope_mpkm,
                    "elevation_mean": self.safe_get(row, "mean.elevation", 100000) / 100,
                    "elev_mean": self.safe_get(row, "mean.elevation", 100000) / 100,
                }
            )

        elif model == "NOAH-OWP-M" or model == "noahowp":
            substitutions.update(
                {
                    "terrain_slope": self.safe_get(row, "mean.slope_1km", 0.01),
                    "azimuth": self.safe_get(row, "circ_mean.aspect", 180),
                    "ISLTYP": int(self.safe_get(row, "mode.ISLTYP", 1)),
                    "IVGTYP": int(self.safe_get(row, "mode.IVGTYP", 1)),
                }
            )

        # Apply substitutions
        try:
            config = template.format(**substitutions)
        except KeyError as e:
            logger.warning(f"Missing key {e} for model {model}, using partial substitution")
            # Try partial substitution
            config = template
            for key, value in substitutions.items():
                config = config.replace(f"{{{key}}}", str(value))

        return config.encode("utf-8")

    def get_file_content(self, path: str) -> Optional[bytes]:
        """Get content for a virtual file"""
        # Check cache first
        if path in self.config_cache:
            return self.config_cache[path]

        parts = path.strip("/").split("/")

        if len(parts) != 2:
            return None

        model = parts[0]
        filename = parts[1]

        # Check if this is a valid model
        if model not in self.models:
            return None

        # Extract divide_id from filename
        _, expected_ext = self.models[model]
        if filename.endswith(expected_ext):
            divide_id = filename[: -len(expected_ext)]
        else:
            return None

        content = self.generate_config(model, divide_id)

        # Cache the content
        # if content:
        #     self.config_cache[path] = content

        return content

    def getattr(self, path):
        """Get file attributes"""
        st = MyStat()

        if path == "/":
            st.st_mode = stat.S_IFDIR | 0o755
            st.st_nlink = 2 + len(self.models)
        elif path.strip("/") in self.models:
            st.st_mode = stat.S_IFDIR | 0o755
            st.st_nlink = 2
        else:
            # Check if it's a config file
            content = self.get_file_content(path)
            if content is not None:
                st.st_mode = stat.S_IFREG | 0o444
                st.st_nlink = 1
                st.st_size = len(content)
            else:
                return -errno.ENOENT

        return st

    def readdir(self, path, offset):
        """Read directory contents"""
        if path == "/":
            yield fuse.Direntry(".")
            yield fuse.Direntry("..")
            for model in self.models:
                yield fuse.Direntry(model)
        elif path.strip("/") in self.models:
            model = path.strip("/")
            _, extension = self.models[model]
            yield fuse.Direntry(".")
            yield fuse.Direntry("..")
            if self.divide_data:
                for divide_id in self.divide_data:
                    yield fuse.Direntry(f"{divide_id}{extension}")

    def open(self, path, flags):
        """Open a file"""
        content = self.get_file_content(path)
        if content is None:
            return -errno.ENOENT

        accmode = os.O_RDONLY | os.O_WRONLY | os.O_RDWR
        if (flags & accmode) != os.O_RDONLY:
            return -errno.EACCES

        return 0

    def read(self, path, size, offset):
        """Read file contents"""
        content = self.get_file_content(path)
        if content is None:
            return -errno.ENOENT

        slen = len(content)
        if offset < slen:
            if offset + size > slen:
                size = slen - offset
            buf = content[offset : offset + size]
        else:
            buf = b""

        return buf


def main():
    """Main entry point"""
    import argparse

    parser = argparse.ArgumentParser(
        description="Virtual Config Directory FUSE Filesystem",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  %(prog)s /path/to/data_folder

This will:
  - Mount at: /path/to/data_folder/config/fuse_config/
  - Use gpkg: /path/to/data_folder/config/*.gpkg
  - Templates from: ./templates/ (or specify with --templates)

The config files will appear as:
  - /path/to/data_folder/config/fuse_config/CFE/{divide_id}.ini
  - /path/to/data_folder/config/fuse_config/lstm/{divide_id}.yml
  - /path/to/data_folder/config/fuse_config/NOAH-OWP-M/{divide_id}.input

To unmount:
  fusermount -u /path/to/data_folder/config/fuse_config
        """,
    )

    parser.add_argument("target_folder", help="Target data folder")
    parser.add_argument(
        "--templates",
        default="./templates",
        help="Directory containing template files (default: ./templates)",
    )
    parser.add_argument(
        "--start-time", default="2023-01-01 00:00:00", help="Start time (YYYY-MM-DD HH:MM:SS)"
    )
    parser.add_argument(
        "--end-time", default="2023-12-31 23:00:00", help="End time (YYYY-MM-DD HH:MM:SS)"
    )
    parser.add_argument("-f", "--foreground", action="store_true", help="Run in foreground")
    parser.add_argument("-d", "--debug", action="store_true", help="Enable debug logging")

    args, unknown = parser.parse_known_args()

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, force=True)

    # Setup paths
    data_dir = Path(args.target_folder)
    config_dir = data_dir / "config"
    mount_point = config_dir / "fuse_config"
    mount_point.mkdir(parents=True, exist_ok=True)

    # Create and configure the filesystem
    fuse_args = [str(mount_point)]
    if args.foreground:
        fuse_args.append("-f")

    server = ConfigFS(
        version="%prog " + fuse.__version__, usage=ConfigFS.fusage, dash_s_do="setsingle"
    )

    server.parser.parse_args(fuse_args)

    try:
        server.initialize_fs(
            data_dir=args.target_folder,
            start_time=args.start_time,
            end_time=args.end_time,
            template_dir=args.templates,
        )

        print("Virtual config filesystem mounted successfully!")
        print(f"Mount point: {mount_point}")
        print(f"Geopackage: {server.gpkg_path}")
        print(f"Templates: {server.template_dir}")
        print(f"Models available: {', '.join(server.models.keys())}")
        print("\nConfig files are now available at:")
        for model in server.models:
            _, ext = server.models[model]
            print(f"  {mount_point}/{model}/*{ext}")
        print("\nPress Ctrl+C to unmount")

        server.main()

    except KeyboardInterrupt:
        print("\nUnmounting...")
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback

        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
