import yaml
import os
from pathlib import Path
from agent.processor import TaskProcessor
from agent.chain import TaskChain, ChainStep

def get_pdf_files(directory: Path) -> list[Path]:
    """Get all PDF files in the directory."""
    return list(directory.glob("*.pdf"))

def load_tasks_config(tasks_dir: str = './agent/tasks') -> dict:
    """Load all YAML files from the tasks directory into a single config dictionary."""
    tasks_config = {}
    tasks_path = Path(tasks_dir)
    
    for yaml_file in tasks_path.glob('*.yaml'):
        with open(yaml_file, 'r') as f:
            config = yaml.safe_load(f)
            tasks_config.update(config)
    
    return tasks_config

def process_directory(directory: Path, processor: TaskProcessor, tasks_config: dict) -> None:
    """Process a single directory to generate topics JSON."""
    print(f"\nProcessing directory: {directory}")
    
    pdf_files = get_pdf_files(directory)
    if not pdf_files:
        print("No PDF files found in the directory, skipping...")
        return

    try:
        # Create chain for topics generation
        topics_steps = [
            ChainStep(
                name="Generate Topics",
                tasks=["create_topics"],
                input_files=pdf_files,
                expect_json=True
            )
        ]
        
        topics_chain = TaskChain(processor, tasks_config, topics_steps)
        topics_result = topics_chain.run("")  # Empty content since we're using PDFs
        
        if not topics_result:
            raise Exception("Failed to generate topics")
            
        # Save topics to file
        output_file = directory / "topics.json"
        output_file.write_text(topics_result, encoding='utf-8')
        print(f"✔️ Topics saved to: {output_file}")

    except Exception as e:
        print(f"❌ Failed to process directory: {directory}")
        print(f"Error: {str(e)}")

def main():
    # Load tasks configuration
    tasks_config = load_tasks_config()
    
    # Initialize processor
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is required")
    processor = TaskProcessor(api_key=api_key)
    
    # Define base directory and target folders
    base_dir = Path("/content/reinforcement_learning_notes")
    target_folders = [
        "01. Multi-armed Bandits"
    ]
    
    # Process each target directory
    for folder in target_folders:
        directory = base_dir / folder
        if directory.exists():
            process_directory(directory, processor, tasks_config)
        else:
            print(f"Directory not found: {directory}")

if __name__ == "__main__":
    main()