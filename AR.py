import streamlit as st
import os
from openai import OpenAI
from moviepy.video.io.VideoFileClip import VideoFileClip
import json

# Initialize OpenAI Client (Make sure OPENAI_API_KEY is in your environment variables)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

def transcribe_video(video_path):
    """Uses OpenAI Whisper to extract text transcript with timestamps."""
    st.info("📢 Extracting audio and generating transcript via Whisper...")
    with open(video_path, "rb") as audio_file:
        transcript_data = client.audio.transcriptions.create(
            file=audio_file,
            model="whisper-1",
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
    return transcript_data.segments

def find_ai_clips(segments):
    """Sends timestamps and text to GPT to detect the best standalone hooks."""
    st.info("🤖 Analyzing content semantic structures with GPT-4o...")
    
    # Format the transcript for the LLM prompt
    formatted_transcript = ""
    for seg in segments:
        formatted_transcript += f"[{seg['start']:.2f} - {seg['end']:.2f}]: {seg['text']}\n"
        
    prompt = f"""
    You are an expert AI video editor specializing in making short-form clips (TikTok, Shorts, Reels).
    Analyze this video transcript and identify 2-3 of the most engaging, high-retention, or interesting standalone segments.
    Each clip must make sense on its own and have a clear hook or topic.
    Keep each clip length between 15 to 60 seconds.

    Transcript:
    {formatted_transcript}

    Respond ONLY with a valid JSON array matching this schema structure:
    [
      {{"title": "Clip Title", "start": 12.5, "end": 45.2, "reason": "Why this segment works"}}
    ]
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"}
    )
    
    data = json.loads(response.choices.message.content)
    return data.get("clips", data) if "clips" in data else data

def slice_and_format_video(input_path, start_time, end_time, output_path):
    """Cuts, reformats to 9:16 layout, and exports the final clip."""
    with VideoFileClip(input_path) as video:
        # Step 1: Subclip extraction
        new_clip = video.subclip(start_time, end_time)
        
        # Step 2: Auto-reframe to portrait 9:16 (Center Crop)
        w, h = new_clip.size
        target_w = int(h * (9 / 16))
        
        if w > target_w:
            # Crop horizontally centered for landscape inputs
            x_start = (w - target_w) // 2
            final_clip = new_clip.crop(x1=x_start, y1=0, x2=x_start + target_w, y2=h)
        else:
            final_clip = new_clip
            
        # Note: Dynamic burnt-in captions typically require complex subtitle file mappings 
        # or clip text overlays per timestamp (e.g., TextClip in moviepy). 
        # For cross-platform stability in this lightweight script, we handle clean 9:16 portrait reformatting.
        
        final_clip.write_videofile(output_path, codec="libx264", audio_codec="aac")

# --- STREAMLIT UI ---
st.set_page_config(page_title="AI Video Clipper Pro", page_icon="✂️", layout="wide")
st.title("✂️ Portrait 9:16 AI Video Clipper & Reframer")
st.caption("Upload landscape videos to automatically isolate viral moments reformatted for Shorts, Reels, and TikTok.")

# Sidebar API Validation Check
if not os.environ.get("OPENAI_API_KEY"):
    openai_key = st.sidebar.text_input("Enter OpenAI API Key", type="password")
    if openai_key:
        client = OpenAI(api_key=openai_key)
    else:
        st.warning("Please provide an OpenAI API Key in the sidebar to run the app.")

uploaded_file = st.file_uploader("Upload a Video File (.mp4 format preferred)", type=["mp4", "mov", "mkv"])

if uploaded_file and (os.environ.get("OPENAI_API_KEY") or 'openai_key' in locals()):
    temp_dir = "temp_processing"
    os.makedirs(temp_dir, exist_ok=True)
    video_path = os.path.join(temp_dir, uploaded_file.name)
    
    with open(video_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
        
    st.video(video_path)
    
    if st.button("🚀 Process, Reframe & Clip Video"):
        try:
            segments = transcribe_video(video_path)
            suggested_clips = find_ai_clips(segments)
            
            st.success(f"✨ Found {len(suggested_clips)} optimal vertical clips!")
            
            for idx, clip in enumerate(suggested_clips):
                st.subheader(f"🎬 Clip {idx+1}: {clip['title']}")
                st.write(f"⏱️ **Timeframe:** {clip['start']}s - {clip['end']}s | 📐 **Format:** 9:16 Vertical Portrait")
                st.write(f"📝 **AI Logic:** {clip['reason']}")
                
                output_clip_name = f"vertical_clip_{idx+1}_{uploaded_file.name}"
                output_clip_path = os.path.join(temp_dir, output_clip_name)
                
                with st.spinner(f"Rendering Vertical Clip {idx+1}..."):
                    slice_and_format_video(video_path, clip['start'], clip['end'], output_clip_path)
                
                st.video(output_clip_path)
                with open(output_clip_path, "rb") as file:
                    st.download_button(
                        label=f"📥 Download Vertical Clip {idx+1}",
                        data=file,
                        file_name=output_clip_name,
                        mime="video/mp4"
                    )
                st.divider()
                
        except Exception as e:
            st.error(f"An error occurred during processing: {str(e)}")
