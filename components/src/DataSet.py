from lsfb_dataset import Downloader # CORRECT (pour l'ancienne version)

downloader = Downloader(dataset='cont', destination="./destination/folder", include_videos=True)
downloader.download()