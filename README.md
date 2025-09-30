__AI generated README.md__
Technically correct but overselling this prototype.
The performance of this is untested and might be unusably slow with ngen

# Virtual Config Filesystem

A FUSE-based virtual filesystem that dynamically generates model configuration files from templates and geopackage data. Instead of creating thousands of physical config files, they're generated on-demand when accessed.

## Features

- **Zero Disk Usage**: Config files don't actually exist on disk
- **Dynamic Generation**: Files are created from templates when accessed
- **Transparent**: Applications read virtual files as if they were real

## Quick Start

```bash
# Install dependencies
pip install fusepy

# Run with your data folder
python config_fuse.py /path/to/data_folder --foreground

# Files are now available at:
ls /path/to/data_folder/config/fuse_config/CFE/
ls /path/to/data_folder/config/fuse_config/lstm/
ls /path/to/data_folder/config/fuse_config/NOAH-OWP-M/
```

## Directory Structure

```
data_folder/
├── config/
│   ├── *.gpkg                  # Your geopackage file (auto-detected)
│   └── fuse_config/            # Mount point (created automatically)
│       ├── CFE/
│       │   └── {divide_id}.ini
│       ├── lstm/
│       │   └── {divide_id}.yml
│       └── NOAH-OWP-M/
│           └── {divide_id}.input

templates/                       # Template directory
├── cfe_template.ini
├── lstm_template.yml
└── noahowp_template.input
```

## Usage

### Basic Usage

```bash
# Mount the filesystem
python main.py /path/to/data_folder

# Use the config files in your application
cat /path/to/data_folder/config/fuse_config/CFE/cat-01.ini

# Unmount when done
fusermount -u /path/to/data_folder/config/fuse_config
```

### Command Line Options

```bash
python main.py <data_folder> [options]

Arguments:
  data_folder          Target data directory containing config/*.gpkg

Options:
  --templates DIR      Template directory (default: ./templates)
  --start-time TIME    Simulation start time (default: 2023-01-01 00:00:00)
  --end-time TIME      Simulation end time (default: 2023-12-31 23:00:00)
  --foreground, -f     Run in foreground (recommended for testing)
  --debug, -d          Enable debug logging
```

## Adding New Models

1. Create a template file in `templates/` with the naming pattern `{model}_template.{extension}`:

```bash
# Example: Add a new "PET" model
cat > templates/pet_template.json << EOF
{
  "catchment": "{divide_id}",
  "area_km2": {area_sqkm},
  "latitude": {lat},
  "longitude": {lon}
}
EOF
```

2. Restart the filesystem - the new model directory will appear automatically:

```bash
python config_fuse.py /path/to/data_folder
ls /path/to/data_folder/config/fuse_config/pet/
# Shows all {divide_id}.json files
```

## Template Variables

Common variables available in templates:

- `{divide_id}` - Catchment/divide identifier
- `{area_sqkm}` or `{areasqkm}` - Catchment area
- `{lat}` or `{latitude}` - Latitude
- `{lon}` or `{longitude}` - Longitude
- `{start_datetime}` - Start time (YYYYMMDDhhmm format)
- `{end_datetime}` - End time (YYYYMMDDhhmm format)

Plus any attributes from your geopackage's `divide-attributes` table.

## Requirements

- FUSE kernel module (usually pre-installed on Linux)
- Optional: pyproj (for coordinate transformations)

## Troubleshooting

**Transport endpoint is not connected**: The filesystem crashed. Unmount and remount:
```bash
fusermount -u /path/to/data_folder/config/fuse_config
```

## How It Works

1. Reads catchment data from the geopackage file
2. Loads template files from the templates directory
3. When a file is accessed, generates content by filling the template with catchment data
4. Returns the generated content as if it were a real file
5. Optionally caches generated files in memory for performance
