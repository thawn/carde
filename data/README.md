# Dataset for the Manuscript: **Data-efficient U-Net for Segmentation of Carbide Microstructures in SEM Images of Steel Alloys**

This dataset contains scanning electron microscopy (SEM) images of steel alloys, including paired secondary electron (SE2) and in-lens (InLens) channels, with corresponding binary segmentation labels. The data supports full reproduction of results presented in the referenced manuscript.

---

## Dataset Description

- **Content:** 13 pairs of SEM images of two reactor pressure vessel (RPV) steels:
  - *JFL*: IAEA reference RPV base metal steel
  - *ANP-10*: Western type RPV steel
- **Acquisition:**  
  - *JFL*: Zeiss NVision 40 microscope  
  - *ANP-10*: Zeiss Ultra 55 microscope  
  - Both SE and InLens detectors used simultaneously.
- **Resolution:** 2048 × 1404 pixels per image  
  - 2048 px width corresponds to 14.3 µm (JFL) or 11.5 µm (ANP-10).

---

## Downloading the dataset

The dataset can be [downloaded from Zenodo](https://zenodo.org/records/16996272?preview=1&token=eyJhbGciOiJIUzUxMiIsImlhdCI6MTc1NjQ5MjQ2NywiZXhwIjoxNzY0NTQ3MTk5fQ.eyJpZCI6IjQzYmYyMTc5LTAwMTctNDYwNi04Y2VhLWU0ZWU4ZjZmMTg1ZiIsImRhdGEiOnt9LCJyYW5kb20iOiI4MDQ4M2Y2M2M2MTQxY2ViMWY1NmI5ODI2ZTYyMzFlZCJ9._LIeCnWSntclqaI0tfmyhOV1ICIDnYzh5z668cGa4ckRqqlq9EIyZ1YjFjwVWx07W7HG0EJARZY5QzKrj-3ToQ).

Download the zip file into the `data/` subdirectory of the code repository and extract the archive:

```bash
cd data/
unzip data.zip
```


## Dataset Structure

These directories contain the relevant data for the manuscript:

```
cloud/
├-─ preprocessed/
│   ├── hold-out/
│   ├── images/
│   └── labels/
├── processed_tiles/
│   ├── images/
│   └── labels/
├── tb_logs/
│   ├── unet_model/
```

### Preprocessed

pre-processed whole images and corresponding labels

### Processed Tiles

tiled images and labels

### tb_logs

trained model
