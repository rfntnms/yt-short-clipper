import gradio as gr
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_video(youtube_url, ai_provider, provider_model):
    logger.info(f"Received processing request for {youtube_url} using {ai_provider}/{provider_model}")
    # Placeholder for actual pipeline logic
    progress_msg = f"Simulating processing for {youtube_url} using {provider_model}..."
    return progress_msg, None

def create_ui():
    with gr.Blocks(title="YT-Short-Clipper v2") as app:
        gr.Markdown("# YT-Short-Clipper (v2 Migration)")
        
        with gr.Tab("Process Video"):
            youtube_url = gr.Textbox(label="YouTube URL", placeholder="https://www.youtube.com/watch?v=...", lines=1)
            
            with gr.Row():
                ai_provider = gr.Dropdown(choices=["OpenAI", "Local LLM"], value="OpenAI", label="AI Provider")
                provider_model = gr.Textbox(label="Provider Model", value="gpt-4", placeholder="gpt-4, llama3, etc.", lines=1)
                
            start_btn = gr.Button("Start Processing", variant="primary")
            
            progress_output = gr.Textbox(label="Processing Status", interactive=False, lines=2)
            video_output = gr.Video(label="Output Video")
            
            start_btn.click(
                fn=process_video,
                inputs=[youtube_url, ai_provider, provider_model],
                outputs=[progress_output, video_output]
            )
            
        with gr.Tab("Settings"):
            gr.Markdown("### Configuration Settings")
            gr.Textbox(label="Base URL", value="https://api.openai.com/v1", lines=1)
            gr.Textbox(label="API Key", type="password", placeholder="sk-...", lines=1)
            
        with gr.Tab("Logs"):
            gr.Markdown("### Application Logs")
            gr.Textbox(label="Log Output", value="Application started...", interactive=False, lines=10)

    return app

if __name__ == "__main__":
    app = create_ui()
    app.launch(server_name="0.0.0.0", server_port=7860)