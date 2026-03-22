# Metashape Spherical → Cubemap → PostShot Converter

> **Fork of [MaikiOS/Agisoft_metashape_convert_to_cubemap](https://github.com/MaikiOS/Agisoft_metashape_convert_to_cubemap)**
> by [smert999](https://github.com/MaikiOS)
>
> This fork adds PostShot-ready COLMAP export and a Japanese UI.
> The original work — spherical-to-cubemap projection math, overlap handling, and Metashape integration — is entirely by smert999.

---

## What this fork adds

| Feature | Original | This fork |
|---|---|---|
| Cubemap conversion | Yes | Yes |
| COLMAP binary export | Partial | Full (PostShot-ready) |
| PostShot drag-and-drop | No | Yes |
| UI language | Russian | Japanese |
| Incomplete cubemap filtering | No | Yes (auto-exclude) |

The new script `postshot_converter.py` is based on `unified_fixed_v002.py` and produces output that can be directly loaded into [PostShot](https://postshot.app/) for 3D Gaussian Splatting.

## Quick start

### Prerequisites

- Agisoft Metashape Professional 1.8+
- OpenCV (installed in Metashape's Python environment)

### Workflow

1. Import spherical images into Metashape
2. Run **Align Cameras** (once)
3. **Tools → Run Script** → select `postshot_converter.py`
4. Choose an output folder, adjust settings, and start processing
5. Drag the output folder into **PostShot**

### Settings

| Setting | Default | Description |
|---|---|---|
| Overlap | 10° | Overlap angle between cubemap faces |
| Face size | Auto | Per-face resolution (1024 / 2048 / 4096 px, or auto) |
| Point limit | 50,000 | Max sparse point cloud points |

## Output structure

```
output_folder/
├── images/           # Cubemap face images (6 faces × N cameras)
├── sparse/0/         # COLMAP binary data
│   ├── cameras.bin   # Intrinsics (PINHOLE model)
│   ├── images.bin    # Extrinsics (position & orientation)
│   └── points3D.bin  # Colored sparse point cloud
└── README.txt        # Processing parameters log
```

## File overview

| File | Description |
|---|---|
| `postshot_converter.py` | **Main script — use this** |
| `unified_fixed_v002.py` | Base script by smert999 (fixed projection math + COLMAP) |
| `convert_to_cubemap_v012.py` | Full GUI version by smert999 |
| `convert_to_cubemap_v011.py` | Earlier version by smert999 |
| `convert_to_cubemap_v009.py` | Earlier version by smert999 |
| `convert_to_cubemap_v007.py` | Earlier version by smert999 |

## Credits

This project would not exist without the excellent foundational work by **smert999** ([@MaikiOS](https://github.com/MaikiOS)):

- Spherical-to-cubemap projection with configurable overlap
- Metashape camera parameter extraction and virtual camera creation
- COLMAP binary format export pipeline
- Critical projection math fixes in `unified_fixed_v002.py`

Thank you for making this work open source.

## License

MIT License — see [LICENSE.md](LICENSE.md)

Original work copyright (c) 2025 smert999.
Fork additions copyright (c) 2026 makotofalcon.
