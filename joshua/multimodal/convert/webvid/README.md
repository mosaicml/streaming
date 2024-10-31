# WebVid

[WebVid](https://m-bain.github.io/webvid-dataset/) is a large-scale dataset of short videos with textual descriptions sourced from stock footage sites. There are two sets of the dataset available:
1. 10M: It contains 10 million video-text pairs.
2. 2.5M: It contains 2.5 million video-text pairs.

## Dataset preparation

To use Streaming Dataset, we must first convert the dataset from its native format to MosaicML's Streaming Dataset format called Mosaic Dataset Shard (MDS). Once in MDS format, we can access the dataset from the local file system (disk network attached storage, etc.) or object store (GCS, OCS, S3, etc.). From the object store, data can be streamed to train deep learning models, and it all works efficiently.

Check out the steps below for information on converting WebVid datasets to MDS format—also, checkout [MDSWriter](https://joshua.docs.mosaicml.com/en/latest/api_reference/generated/joshua.MDSWriter.html) parameters for details on advanced usage.

#### Single MDS dataset conversion

Create an MDS dataset from a CSV file containing video URLs (downloads the videos).

1. Navigate to the [WebVid download section](https://m-bain.github.io/webvid-dataset/), where you will find 2.5M and 10M dataset splits. Download each CSV split you want to process.
2. Run the [crawl_webvid.py](https://github.com/mosaicml/joshua/blob/main/joshua/multimodal/convert/webvid/crawl_webvid.py) script with minimum required arguments as shown below  
    <!--pytest.mark.skip--> 
    ```
    python crawl_webvid.py --in <CSV filepath> --out_root <Output MDS directory> 
    ```
#### Multiple MDS sub-dataset conversion

Create multiple MDS sub-datasets from a CSV file containing video URLs and a list of substrings to match against (downloads the videos).

1. Navigate to the [WebVid download section](https://m-bain.github.io/webvid-dataset/), where you will
   find 2.5M and 10M dataset splits. Download each CSV split you want to process.

2. Run the [crawl_webvid_subsets.py](https://github.com/mosaicml/joshua/blob/main/joshua/multimodal/convert/webvid/crawl_webvid_subsets.py) script with minimum required arguments as shown below. The script also supports an optional arg `filter`, which takes a comma-separated list of keywords to filter into sub-datasets.
    <!--pytest.mark.skip-->
    ```
    python crawl_webvid_subsets.py --in <CSV filepath> --out_root <Output MDS directory>
    ```

#### Split out MDS datasets column
Iterate an existing MDS dataset containing videos, creating a new MDS dataset without video contents embedded in it, instead, add a video filepath in a new MDS dataset where the video files (MP4) are stored separately.

1. Run the [extract_webvid_videos.py](https://github.com/mosaicml/joshua/blob/main/joshua/multimodal/convert/webvid/extract_webvid_videos.py) script with minimum required arguments as shown below
    <!--pytest.mark.skip-->
    ```
    python extract_webvid_videos.py --in <Input mp4-inside MDS dataset directory> --out_mds <Output mp4-outside MDS dataset directory> --out_mp4 <Output mp4 videos directory>
    ```
