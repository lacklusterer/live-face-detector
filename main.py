from helpers.video import * 

def main():
    """Entry point of the script."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time video processor with face detection")
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--fps', type=float, default=30.0, help='Target FPS')
    parser.add_argument('--width', type=int, default=640, help='Camera width')
    parser.add_argument('--height', type=int, default=480, help='Camera height')
    parser.add_argument(
        '--model', 
        choices=['haarcascade', 'yunet'], 
        default='yunet', 
        help="Model to use for face detection (default: yunet)"
    )
    
    args = parser.parse_args()
    
    config = VideoConfig(
        fps=args.fps,
        width=args.width,
        height=args.height,
        model=args.model
    )
    
    processor = VideoProcessor(config, debug=args.debug)
    processor.run()

if __name__ == "__main__":
    main()
