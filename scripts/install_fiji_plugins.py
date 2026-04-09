"""Download Fiji plugin jars required by CollagenFeatures."""

from __future__ import annotations

from pathlib import Path
from urllib.request import urlretrieve


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DESTINATION = PROJECT_ROOT / "third_party" / "twombli"

PLUGIN_URLS = {
    "OrientationJ_.jar": "https://bigwww.epfl.ch/demo/orientationj/OrientationJ_.jar",
    "TWOMBLI_.jar-20250311140257": "https://sites.imagej.net/TWOMBLI/jars/TWOMBLI_.jar-20250311140257",
    "anamorf-3.1.0.jar-20240717211407": "https://sites.imagej.net/TWOMBLI/plugins/anamorf-3.1.0.jar-20240717211407",
    "iaclasslibrary-1.0.32.jar-20240717211407": "https://sites.imagej.net/TWOMBLI/plugins/iaclasslibrary-1.0.32.jar-20240717211407",
    "ij_ridge_detect-1.4.0.jar-20170820141758": "https://sites.imagej.net/Biomedgroup/plugins/ij_ridge_detect-1.4.0.jar-20170820141758",
    "Maximum_Inscribed_Circle.jar-20150820132158": "https://biop.epfl.ch/Fiji-Update/plugins/BIOP/Maximum_Inscribed_Circle.jar-20150820132158",
    "commons-csv-1.10.0.jar": "https://repo1.maven.org/maven2/org/apache/commons/commons-csv/1.10.0/commons-csv-1.10.0.jar",
}


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    for filename, url in PLUGIN_URLS.items():
        destination = DESTINATION / filename
        if destination.exists():
            print(f"Skipping existing {filename}")
            continue
        print(f"Downloading {filename}")
        urlretrieve(url, destination)
    print(f"Downloaded Fiji plugin dependencies to {DESTINATION}")


if __name__ == "__main__":
    main()
