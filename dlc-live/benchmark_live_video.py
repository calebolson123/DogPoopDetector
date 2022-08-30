import dlclive

dlclive.benchmark_videos(
    'path_to_model',
    'path_to_video',
    resize=0.75,
    display=True,
    pcutoff=0.5,
    display_radius=2,
    cmap='bmy')