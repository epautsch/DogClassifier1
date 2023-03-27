from argparse import ArgumentParser, Namespace


def detection() -> Namespace:
    parser: ArgumentParser = ArgumentParser(
        prog='Dog Breed Classifier and Detector',
        description='A tool for classifying and detecting dogs in images based on 120 different breeds',
    )

    parser.add_argument(
        '-i', '--image',
        help='Path to the image file',
        type=str,
        required=False,
        default=None,
    )

    parser.add_argument(
        '-o', '--output',
        help='Path to the output image file. DEFAULT: "./<input image name>_predict.png"',
        type=str,
        required=False,
        default=None,
    )

    return parser.parse_args()

