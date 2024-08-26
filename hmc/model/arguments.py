import argparse


def build():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        '--job-dir',
        help='jobs staging path on google cloud storage',
        required=True
    )

    ap.add_argument(
        '--job-name',
        help='set job name',
        required=True
    )

    ap.add_argument(
        '--bucket-name',
        help='gcs bucket name',
        required=True
    )

    ap.add_argument(
        '--train-path',
        help='gs://.../',
        required=True
    )

    ap.add_argument(
        '--trainset-pattern',
        help='gs://.../',
        required=True
    )

    ap.add_argument(
        '--validationset-pattern',
        help='gs://.../',
        required=True
    )

    ap.add_argument(
        '--testset-path',
        help='gs://../testset.pk',
        required=True
    )

    ap.add_argument(
        '--metadata-path',
        help='gs://../metadata.json',
        required=True
    )

    ap.add_argument(
        '--labels-path',
        help='gs://../labels.json',
        required=True
    )

    ap.add_argument(
        '--batch-size',
        default=32,
        type=int
    )

    ap.add_argument(
        '--epochs',
        default=10,
        type=int
    )

    ap.add_argument(
        '--base-path',
        required=True,
        type=str
    )

    return ap.parse_args()

