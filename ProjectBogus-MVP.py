#!/usr/bin/env python
# coding: utf-8

import math
import subprocess
import torch
import torchaudio
import torchvision
import accelerate
import ollama
import json
import sys
import os
import numpy as np
import argparse
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from pyannote.core import Annotation
from moviepy.editor import TextClip, CompositeVideoClip, VideoFileClip, ColorClip
from datasets import load_dataset
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


def main():


    arg_parser = argparse.ArgumentParser()

    # CLI argument parsing

    arg_parser.add_argument("-u", "--url", help="The Video URL")
    arg_parser.add_argument("-df", "--downloadfolder",nargs='?', const="downloaded", help="Folder in which to download video",default="downloaded")
    arg_parser.add_argument("-v", "--videoname",nargs='?', const="bogus_vid", help="Downloaded video name", default="bogus_vid")
    arg_parser.add_argument("-if", "--inputfolder",nargs='?', const="downloaded", help="Folder where the downloaded video is", default="downloaded")
    arg_parser.add_argument("-of", "--outputfolder",nargs='?', const="transcripted_videos", help="Folder where the cut videos are outputted", default="transcripted_videos")
    arg_parser.add_argument("-ar", "--aspectratio",nargs='?', const=(9,16), help=" Aspect Ratio", type=tuple, default=(9,16))
    arg_parser.add_argument("-ts", "--transcriptionid",nargs='?', const="openai/whisper-large-v3", help="Model for initial video transcription", default="openai/whisper-large-v3")
    arg_parser.add_argument("-tt", "--subtitlesid",nargs='?', const="openai/whisper-large-v3", help="Model for shorts video transcription with word-level timestamps", default="openai/whisper-large-v3")
    arg_parser.add_argument("-ms", "--maxsize",nargs='?', const=550, help="Max size in seconds of each transcript chunk; Should not exceed 620", type=int, default=550)
    arg_parser.add_argument("-hf", "--hftoken",nargs='?', const="hf_XXXXXXX", help="HugginFace token used for the diarization model",
                             default="hf_XXXXXXXXXX")
    arg_parser.add_argument("-is", "--inputsubs",nargs='?', const="transcripted_videos", help="The folder in which to look for the cut videos", default="transcripted_videos")
    arg_parser.add_argument("-os", "--outputsubs",nargs='?', const="subtitled_videos", help="The folder in which to look for the cut videos", default="subtitled_videos")
    arg_parser.add_argument("-m", "--metafolder",nargs='?', const="metadata", help="The path to the metadata files", default="metadata")
    arg_parser.add_argument("-l", "--lang",nargs='?', const="english", help="Transcription language", default="english")
    arg_parser.add_argument("-rc", "--regencount",nargs='?', const=2, help="Amount of LLM regenerations per chunk",type=int, default=2)
    arg_parser.add_argument("-ln", "--llmname",nargs='?', const="llama3", help="The LLM model's name, ex. gemma:7b", default="llama3")

    args = arg_parser.parse_args()

    video_url = args.url
    video_name = args.videoname
    download_folder = args.downloadfolder
    input_folder = args.inputfolder
    output_folder = args.outputfolder
    target_aspect_ratio = args.aspectratio
    transcription_model_id = args.transcriptionid
    max_size = args.maxsize
    language = args.lang
    hf_token = args.hftoken
    input_folder_subs = args.inputsubs
    output_folder_subs = args.outputsubs
    transcription_subs_model_id = args.subtitlesid
    metadata_folder = args.metafolder
    max_regen_count = args.regencount
    llm_model_name = args.llmname

    branding = "bogus"

    log_path = os.path.join(metadata_folder,"latest_log.txt")

    # Ensure dirs exist
    os.makedirs(output_folder, exist_ok=True)
    os.makedirs(output_folder_subs, exist_ok=True)
    os.makedirs(download_folder, exist_ok=True)
    os.makedirs(metadata_folder, exist_ok=True)

    ### Here we call subprocess to execute youtube download via the command line

    with open(log_path, "w") as log_writer:
        log_writer.write(f"[{branding}] Downloading video")
    print(f"[{branding}] Downloading video")

    command = [
        "yt-dlp",
        "-x", # Extract audio
        "--audio-format", "wav",
        "--audio-quality", "0", # Best quality
        video_url,
        "-o", f"{download_folder}/{video_name}.wav",
        "--force-overwrites", # Always redownload the video; This is because the downloaded video always has the same name
        "-k", # Keep the video
        "-f", # Extra parameters below
        """(bestvideo[ext=mp4][height<=1080][protocol!=http_dash_segments]+bestaudio[ext=m4a]/bestvideo[ext=webm][height<=1080][protocol!=http_dash_segments]
        +bestaudio[ext=webm])/(best[ext=mp4][height<=1080][protocol!=http_dash_segments]/best[ext=webm][height<=1080][protocol!=http_dash_segments])""",
        "--merge-output-format", "mp4" # Download 1080p or below to avoid excessive sizes; Prefer wav audio
    ]

    ytdl = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    with open(log_path, "w") as log_writer:
        log_writer.write("\n" + ytdl.stdout)
        log_writer.write("\n" + ytdl.stderr)
    print("\n" + ytdl.stdout)
    print("\n" + ytdl.stderr)

    ### Initial transcription and diarization of the extracted audio from the downloaded video
    with open(log_path, "a") as log_writer:
        log_writer.write(f"\n[{branding}] Initial transcription and diarization of the extracted audio from the downloaded video")
    print(f"\n[{branding}] Initial transcription and diarization of the extracted audio from the downloaded video")

    pipelineD = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1",
    use_auth_token=hf_token)
    pipelineD.to(torch.device("cuda"))

    waveform, sample_rate = torchaudio.load(f"{input_folder}/{video_name}.wav")
    with ProgressHook() as hook:
        diarization = pipelineD({"waveform": waveform, "sample_rate": sample_rate}, hook=hook)
    assert isinstance(diarization, Annotation)

    # End of diarization code
    # Begins transcription code
    with open(log_path, "a") as log_writer:
        log_writer.write(f"\n[{branding}] Begins transcription code")
    print(f"\n[{branding}] Begins transcription code")


    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

    model_id_TM = transcription_model_id

    model_TM = AutoModelForSpeechSeq2Seq.from_pretrained(model_id_TM, torch_dtype=torch_dtype,
                                                    low_cpu_mem_usage=False, # Change if RAM use is excessive
                                                    use_safetensors=True,
                                                    use_flash_attention_2=True) # Modern GPU optimization; May not work on older graphics processors.

    model_TM.to(device)

    processor_TM = AutoProcessor.from_pretrained(model_id_TM)

    pipelineTM = pipeline(
        "automatic-speech-recognition",
        model=model_TM,
        tokenizer=processor_TM.tokenizer,
        feature_extractor=processor_TM.feature_extractor,
        max_new_tokens=128,
        chunk_length_s=30,
        batch_size=16,
        torch_dtype=torch_dtype,
        device=device,
    )

    result = pipelineTM(f"{input_folder}/{video_name}.wav", generate_kwargs={"language": language}, return_timestamps=True)

### End of transcription and diarization code; Combining diarization and transcription into one by matching the timestamps of each
    with open(log_path, "a") as log_writer:
        log_writer.write(f"\n[{branding}] Combining data")
    print(f"\n[{branding}] Combining data")

    combined_data = []

    current_speaker = None

    for sentence_data in result["chunks"]:
        start_time, end_time = sentence_data['timestamp']

        for speech_turn, track, speaker in diarization.itertracks(yield_label=True):
            if start_time <= speech_turn.end:
                current_speaker = speaker
                break

        sentence_text = sentence_data['text']

        if(end_time == None):
            # If the transcription model did not detect an ending timestamp we need to add it from the diarization model's detection
            end_time = round(speech_turn.end,2)
            timestamp_list = list(result["chunks"][-1]["timestamp"])
            timestamp_list[1] = round(speech_turn.end,2)
            timestamp_list = tuple(timestamp_list)
            result["chunks"][-1]["timestamp"] = timestamp_list
        combined_sentence = f'({start_time}s-{end_time}s)[{current_speaker}] \"{sentence_text.strip()}\"'
        combined_data.append(combined_sentence)

    combined_output = ",\n".join(combined_data)

    ### Splitting the transcription into parts dynamically based on length
    with open(log_path, "a") as log_writer:
        log_writer.write(f"\n[{branding}] Splitting transcript")
    print(f"\n[{branding}] Splitting transcript")
    parts = combined_output.split(',\n')

    num_parts = len(parts)
    sec_length = result["chunks"][-1]["timestamp"][1]
    # If the ratio between the ending timestamp of the transcript and the maximum size of a chunk is above 1.13, it indicates that the current transcript is too long
    if (sec_length / max_size) < 1.13:
        split_parts = 1
    elif (sec_length / max_size) < 2:
        split_parts = 2
    elif (sec_length / max_size) >= 2:
        # We round up the parts required
        split_parts = (math.ceil(sec_length / max_size))
        
    split_size = num_parts // split_parts

    result_parts = [parts[i:i+split_size] for i in range(0, num_parts, split_size)]

    if len(result_parts) > split_parts:
        result_parts[-2].extend(result_parts[-1])
        del result_parts[-1]

    results = [',\n'.join(part) for part in result_parts]
    with open(log_path, "a") as log_writer:
        log_writer.write(f"\n[{branding}] Acquiring video duration")
    print(f"\n[{branding}] Acquiring video duration")
    # We need the video duration to perform a timestamp check on the LLM outputs

    def get_video_duration(video_path):
        try:
            # Run ffprobe command to get duration in JSON format
            result = subprocess.run(
                [
                    'ffprobe',
                    '-v', 'error',
                    '-select_streams', 'v:0',
                    '-show_entries', 'format=duration',
                    '-of', 'json',
                    video_path
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=True,
                text=True
            )

            # Parse the JSON output
            probe_result = json.loads(result.stdout)
            with open(log_path, "a") as log_writer:
                log_writer.write("\n" + result.stdout)
                log_writer.write("\n" + result.stderr)
            print("\n" + result.stdout)
            print("\n" + result.stderr)
            duration = float(probe_result['format']['duration'])
            return duration

        except (subprocess.CalledProcessError, KeyError, ValueError, json.JSONDecodeError) as e:
            with open(log_path, "a") as log_writer:
                log_writer.write(f"An error occurred: {e}")
            print(f"An error occurred: {e}")
            return None

    video_path = os.path.join(input_folder,f"{video_name}.mp4")
    video_duration = get_video_duration(video_path)

    with open(log_path, "a") as log_writer:
        log_writer.write(f"\n[{branding}] Ollama starting")
    print(f"\n[{branding}] Ollama starting")

    ### Large Language Model transcript processing

    # Initialize result list
    res = []
    chunk_errors = []

    temperature = 2.5

    for index, part in enumerate(results):
            regen_count = 0
            prompt = f"""
    Generate multiple valid JSON data entries containing stories not shorter than 20 seconds extracted from the provided dialogue '{part}'.

    **Instructions:**
    - Pick the most interesting parts of the video - the parts you picked are further referred to as "story" or "stories".
    - Each story must be at least 20 seconds long.
    - Ensure at least one story per entry and avoid empty responses.
    - Ensure each story is interesting and compelling.
    - The time format must remain consistent.That means to make the story duration accurate.
    - Instructions to follow - story["end_time"] >= story["start_time"]: - story["end_time"] < video_duration: - 
    - Follow the provided instructions precisely.

    **JSON Structure:**
    Each JSON entry must follow this structure:

    {{
    "stories": [
        {{
        "title": "GENERATED_TITLE",
        "description": "GENERATED_DESCRIPTION",
        "start_time": FLOAT_START_TIME,
        "end_time": FLOAT_END_TIME,
        "tags": [GENERATED_TAGS],
        "virality": INT_VIRALITY_SCORE
        }}
    ]
    }}

    **Story Duration:**
    - Each story should be at least 20 seconds long calculated using the difference between end timestamp and start timestamp.
    - Story start time is the start time of the sentence the story begins with (first sentence of the story you picked), that is not 0. Example: (1816.96s-1817.38s)[SPEAKER_01] 'What do you have?', if that is the first sentence of the story, the story start time is 1816.96, not 0. This is very crucial to follow.
    - Story end time is the end time of the sentence the story ends with (last sentence of the story you picked), that is not 0. Example: (1816.96s-1817.38s)[SPEAKER_01] 'What do you have?', if that is the last sentence of the story, the story end time is 1816.96, not 0. This is very important to follow.
    - The previous two sentences were just examples, do not use them in your responses
    - Do not include ads as stories

    **Virality:**
    - Only include stories with a virality score above 40/100.

    **Quotes:**
    - When quoting a word as-is, avoid surrounding it with quotation marks.

    **Avoid:**
    - Using the provided example stories in the response.
    - Using a JSON key called story; call the top-level JSON key stories.
    - Returning a nonsensical array of ints.

    **Ensure:**
    - Each story within an entry has a unique title, description, tags, and virality score.
    - Each story is at least 20 seconds long between starting and ending timestamp.
    - All responses are in valid JSON format, and each key-value pair is properly opened and closed.
    - There are no abrupt endings or incomplete stories.

    **Example Dialogue Format:**
    - "(1816.96s-1817.38s)[SPEAKER_01] 'What do you have?'"
    - "(1817.88s-1819.44s)[SPEAKER_03] 'I say 20.1%.'"
    """
            with open(log_path, "a") as log_writer:
                log_writer.write(f"\n[{branding}] Begin generation")
            print(f"\n[{branding}] Begin generation")
            while True:

                try:
                    gen = ollama.generate(model=f'{llm_model_name}', prompt=prompt, context=[], format="json", stream=False, options={
            "num_batch": 2096,
            "num_gpu": 1,
            "main_gpu": "GPU-XXXXXX",                                                                                                               
            "low_vram": False,
            "temperature": temperature,
            "top_p": 0.99, 
            "top_k": 100, 
            "num_ctx": 8000, 
            "num_predict": 1500})
                    with open(log_path, "a") as log_writer:
                        log_writer.write(f"\n[{branding}] Now printing for part {index + 1}/{len(results)}\n")
                        log_writer.write(gen["response"])
                    print((f"\n[{branding}] Now printing for part {index + 1}/{len(results)}\n"))
                    print(gen["response"])

                    json_response = json.loads(gen["response"])
                    time_diff = []

                    for story in json_response["stories"]:
                        if story["end_time"] <= story["start_time"]:
                            with open(log_path, "a") as log_writer:
                                log_writer.write(f"Story start {story['start_time']}")
                                log_writer.write(f"Story end {story['end_time']}")
                            print(f"Story start {story['start_time']}")
                            print(f"Story end {story['end_time']}")
                            break

                        if story["end_time"] > video_duration:
                            with open(log_path, "a") as log_writer:
                                log_writer.write(f"Story end {story['end_time']}")
                            print(f"Story end {story['end_time']}")
                            break

                        if story["title"] == "":
                            with open(log_path, "a") as log_writer:
                                log_writer.write(f"Story title {story['title']}")
                            print(f"Story title {story['title']}")
                            break
                        
                        if story["description"] == "":
                            with open(log_path, "a") as log_writer:
                                log_writer.write(f"Story title {story['description']}")
                            print(f"Story title {story['description']}")
                            break

                        if index > 0 and story["start_time"] < 400:
                            with open(log_path, "a") as log_writer:
                                log_writer.write(f"Story start {story['start_time']}")
                            print(f"Story start {story['start_time']}")
                            break
                        
                        if index > 0 and story["end_time"] < 400:
                            with open(log_path, "a") as log_writer:
                                log_writer.write(f"Story end {story['end_time']}")
                            print(f"Story end {story['end_time']}")
                            break

                        time_diff.append(story["end_time"] - story["start_time"])

                    if not time_diff:
                        with open(log_path, "a") as log_writer:
                            log_writer.write(f"\n[{branding}] No valid stories in this generation. Regenerating.")
                        print(f"\n[{branding}] No valid stories in this generation. Regenerating.")
                        
                    elif max(time_diff) >= 15:
                        res.append(json_response)
                        break

                    if regen_count >= max_regen_count:
                        with open(log_path, "a") as log_writer:
                            log_writer.write(f"\n[{branding}] Error: Did not generate valid length in an adequate amount of steps for the chunk. Modify the prompt.")
                        print(f"\n[{branding}] Error: Did not generate valid length in an adequate amount of steps for the chunk. Modify the prompt.")
                        chunk_errors.append(index+1)
                        break
                
                    regen_count += 1

                    with open(log_path, "a") as log_writer:
                        log_writer.write(f"\n[{branding}] Did not break loop, re-generating.")
                    print(f"\n[{branding}] Did not break loop, re-generating.")

                except (json.JSONDecodeError, KeyError, Exception) as e:
                    with open(log_path, "a") as log_writer:
                        log_writer.write(f"\n[{branding}] Error encountered: {e}")
                    print(f"\n[{branding}] Error encountered: {e}")
                    continue

    if len(res) < 1:
            with open(log_path, "a") as log_writer:
                log_writer.write(f"\n[{branding}] No valid stories generated, consider re-running, changing the video and seed, or increasing the regeneration count.")
            print(f"\n[{branding}] No valid stories generated, consider re-running, changing the video and seed, or increasing the regeneration count.")

    for out in res:
        with open(log_path, "a") as log_writer:
            log_writer.write(json.dumps(out, indent=2))
        prompt_success_rate = round(((len(results) - len(chunk_errors)) / len(results)) * 100, 2)
        if max_regen_count == 0:
            max_regen_count = 1
    with open(log_path, "a") as log_writer:
        log_writer.write(f"""\n[{branding}] The following chunks failed to generate valid stories in the provided regeneration count: {chunk_errors}.\n
        This equates to a prompt success rate of: {prompt_success_rate}% and {prompt_success_rate/max_regen_count}% adjusted efficiency.""")
    print(f"""\n[{branding}] The following chunks failed to generate valid stories in the provided regeneration count: {chunk_errors}.\n
        This equates to a prompt success rate of: {prompt_success_rate}% and {prompt_success_rate/max_regen_count}% adjusted efficiency.""")

    ### Handling the outputted data from the LLM
    with open(log_path, "a") as log_writer:
        log_writer.write(f"\n[{branding}] Ollama finished")
    print(f"\n[{branding}] Ollama finished")

    json_chunk_array = []
    generation_json = []

    for out in res:
        data = json.loads(json.dumps(out))
        json_chunk_array.append(data)
        generation_json.extend(data['stories'])

    with open(log_path, "a") as log_writer:
        log_writer.write(f"\n[{branding}] Trimming starting")
    print(f"\n[{branding}] Trimming starting")

    # Cutting the transcript and adding the relevant parts to each story with a subtitles key;
    # Extending stories beyond the initial timestamps if the timestamp does not end at a dot or an exclamation mark

    def trim_transcript(transcript, start_time, end_time, story):
        trimmed_transcript = []
        added_sentences = 0
        sentence_index = 0  # Initialize an index to keep track of current sentence position

        # Iterate through each sentence in the transcript
        while sentence_index < len(transcript):
            sentence = transcript[sentence_index]
            sentence_start, sentence_end = sentence['timestamp']
            
            if sentence_start <= end_time and sentence_end >= start_time:
                clip_start_time = max(start_time, sentence_start)
                clip_end_time = min(end_time, sentence_end)
                
                clipped_sentence = {
                    'timestamp': (clip_start_time, clip_end_time),
                    'text': sentence['text']
                }
                trimmed_transcript.append(clipped_sentence)
                added_sentences += 1
                
                # Handle additional sentences if the sentence does not end with punctuation
                if clipped_sentence['text'][-1] not in ['.', '!'] and added_sentences < 10:
                    sentence_index += 1  # Move to the next sentence
                    while sentence_index < len(transcript):
                        next_sentence = transcript[sentence_index]
                        next_sentence_start, next_sentence_end = next_sentence['timestamp']
                        
                        if len(trimmed_transcript) == 1:
                            story["end_time"] = next_sentence_end
                            
                        clipped_next_sentence = {
                            'timestamp': (next_sentence_start, next_sentence_end),
                            'text': next_sentence['text']
                        }
                        trimmed_transcript.append(clipped_next_sentence)
                        added_sentences += 1
                        
                        if next_sentence['text'][-1] in ['.', '!'] or added_sentences >= 10:
                            break
                        sentence_index += 1
                
                if trimmed_transcript[-1]['text'][-1] not in ['.', '!']:
                    trimmed_transcript[-1]['text'] += '.'
                    
                if added_sentences >= 10:
                    break
            
            sentence_index += 1  # Move to the next sentence

        return trimmed_transcript

    transcript = result["chunks"]

    for generation in json_chunk_array:
        for story in generation["stories"]:
            start_time = story["start_time"]
            end_time = story["end_time"]
            story["subtitles"] = trim_transcript(transcript, start_time, end_time, story)
            story["end_time"] = story["subtitles"][-1]["timestamp"][1]


    # Write a json file with all the relevant data from the generation
    with open(log_path, "a") as log_writer:
        log_writer.write(f"\n[{branding}] Writing generation metadata json")
    print(f"\n[{branding}] Writing generation metadata json")

    with open(f'{metadata_folder}/generation_info.json', 'w') as file:
        json.dump(json_chunk_array, file, indent=4)


    ### Here we cut the videos along the LLM provided timestamp

    def get_video_dimensions(video_file):
        command = [
            "ffprobe",
            "-v", "error",
            "-show_entries", "stream=width,height",
            "-of", "json",
            video_file
        ]
        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
        video_info = json.loads(output.decode('utf-8'))
        width = video_info['streams'][0]['width']
        height = video_info['streams'][0]['height']
        print("\n" + output.decode('utf-8'))
        with open(log_path, "a") as log_writer:
            log_writer.write("\n" + output.decode('utf-8'))
        return width, height

    def calculate_crop_dimensions(original_width, original_height, target_aspect_ratio):
        target_width_ratio, target_height_ratio = target_aspect_ratio

        if original_width / original_height > target_width_ratio / target_height_ratio:
            # Crop width
            new_width = int(original_height * (target_width_ratio / target_height_ratio))
            new_height = original_height
            crop_x = (original_width - new_width) // 2
            crop_y = 0
        else:
            # Crop height
            new_width = original_width
            new_height = int(original_width / (target_width_ratio / target_height_ratio))
            crop_x = 0
            crop_y = (original_height - new_height) // 2

        return new_width, new_height, crop_x, crop_y


    # Get video dimensions
    video_file = os.path.join(input_folder, f'{video_name}.mp4')
    original_width, original_height = get_video_dimensions(video_file)

    # Calculate crop dimensions
    new_width, new_height, crop_x, crop_y = calculate_crop_dimensions(original_width, original_height, target_aspect_ratio)

    len_res = len(res)
    completed = 0

    with open(log_path, "a") as log_writer:
        log_writer.write("\n[bogus] Beginning to cut.\n")
    for y in range(len_res):
        json_data = json_chunk_array[y]
        length = len(json_data["stories"])
        completed += 1

        for x in range(length):
            start_time = json_data["stories"][x]["start_time"]
            end_time = json_data["stories"][x]["end_time"]
            time_diff = end_time - start_time

            output_file = f'{output_folder}/{json_data["stories"][x]["title"]}.mp4'

            cmd = [
                "ffmpeg",
                "-y",
                "-ss", str(start_time),
                "-to", str(end_time),
                "-i", video_file,
                "-vf", f"crop={new_width}:{new_height}:{crop_x}:{crop_y},scale='if(gt(iw,ih),iw*(max(iw,ih)/min(iw,ih)),iw*(max(iw,ih)/min(iw,ih)))':'if(gt(iw,ih),ih*(max(iw,ih)/min(iw,ih)),ih*(max(iw,ih)/min(iw,ih)))'",
                "-b:a", "257k",
                "-b:v", "5M",
                "-r", "30",
                "-ar", "48000",
                output_file
            ]
            with open(log_path, "a") as log_writer:
                log_writer.write(f"[bogus] Running command: {' '.join(cmd)}")
                print(f"[bogus] Running command: {' '.join(cmd)}")

            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            o, e = proc.communicate()

            if proc.returncode != 0:
                with open(log_path, "a") as log_writer:
                    log_writer.write(f"[FFmpeg] Error for chunk {completed}/{len_res} with return code {proc.returncode}")
                    log_writer.write(f"[FFmpeg] FFmpeg stdout: {o.decode('utf-8')}")
                    log_writer.write(f"[FFmpeg] FFmpeg stderr: {e.decode('utf-8')}")
                print(f"[FFmpeg] Error for chunk {completed}/{len_res} with return code {proc.returncode}")
                print(f"[FFmpeg] FFmpeg stdout: {o.decode('utf-8')}")
                print(f"[FFmpeg] FFmpeg stderr: {e.decode('utf-8')}")
            else:
                with open(log_path, "a") as log_writer:
                    log_writer.write(f"\n[FFmpeg] Successfully created file: {output_file}")
                    print(f"\n[FFmpeg] Successfully created file: {output_file}")

    with open(log_path, "a") as log_writer:
        log_writer.write(f"\n[{branding}] FFmpeg Done! If no codes have been printed, check if the prompt returns adequate length videos.")
    print(f"\n[{branding}] FFmpeg Done! If no codes have been printed, check if the prompt returns adequate length videos.")

    ### Beginning of subtitling
    with open(log_path, "a") as log_writer:
        log_writer.write(f"\n[{branding}] Now subtitling")
    print(f"\n[{branding}] Now subtitling")

    for video in os.listdir(input_folder_subs):
        if not video.endswith(".mp4"): # Skip everything that is not mp4
            continue
        videofilename = f"{input_folder_subs}/{video}"
        audiofilename = videofilename.replace(".mp4",'.wav')

        ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-i", videofilename,  
        "-vn",                
        "-acodec", "pcm_s16le",
        "-b:a", "257k",
        "-ar", "48000",
        "-q:a", "1",         
        audiofilename         
    ]

        cmd = subprocess.run(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        print("\n" + cmd.stdout)
        print("\n" + cmd.stderr)
        with open(log_path, "a") as log_writer:
            log_writer.write("\n" + cmd.stdout)
            log_writer.write("\n" + cmd.stderr)
        
        
        # We check if the hardware has a captable gpu, else use the processor
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        
        model_id_TS = transcription_subs_model_id
        
        
        # Parameters we pass to the model like using flash attention 2 which if supported by the gpu speeds it up
        modelTS = AutoModelForSpeechSeq2Seq.from_pretrained(model_id_TS, torch_dtype=torch_dtype,
                                                        low_cpu_mem_usage=False,
                                                        use_safetensors=True,
                                                        output_attentions=False,
                                                        use_flash_attention_2=False)
        
        # We send the model to the gpu
        modelTS.to(device)
        
        processorTS = AutoProcessor.from_pretrained(model_id_TS)
        
        # Generation parameters, max new tokens is how many words the ai should return, chunk size is size per word
        pipelineTS = pipeline(
            "automatic-speech-recognition",
            model=modelTS,
            tokenizer=processorTS.tokenizer,
            feature_extractor=processorTS.feature_extractor,
            max_new_tokens=128,
            chunk_length_s=30,
            batch_size=16,
            torch_dtype=torch_dtype,
            device=device,
        )

        dataset = load_dataset("distil-whisper/librispeech_long", "clean", split="validation")
        sample = dataset[0]["audio"]
        
        # Here test.mp3 is where the mp3 input is put
        segments = pipelineTS(audiofilename, generate_kwargs={"language": "english"}, return_timestamps="word")

        def get_video_duration(file_path):
            # Run ffprobe command to get video information
            result = subprocess.run(
                ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'json', file_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE
            )

            # Check if ffprobe was successful
            if result.returncode == 0:
                # Parse the JSON output
                data = json.loads(result.stdout.decode('utf-8'))
                # Get the duration from the JSON data
                duration = float(data['format']['duration'])
                return duration
            else:
                # log_writer.write error message if ffprobe failed
                with open(log_path, "a") as log_writer:
                    log_writer.write("Error:", result.stderr.decode('utf-8'))
                print("Error:", result.stderr.decode('utf-8'))
                return None
        
        wordlevel_info = []
        
        for segment in segments["chunks"]:
            if(segment["timestamp"][1] == None):
                timestamp_list = list(segment["timestamp"])
                timestamp_list[1] = get_video_duration(videofilename)
                timestamp_list = tuple(timestamp_list)
                segment["timestamp"] = timestamp_list
            if segment["text"]:
                if segment["text"][-1] in ["-", "?", ","]:
                    segment["text"] = segment["text"][:-1]
            if segment["text"]:
                if segment["text"][0] in ["-","?",",","."]:
                    segment["text"] = segment["text"][1:]
            if segment["text"]:
                wordlevel_info.append({'word':segment["text"],'start':segment["timestamp"][0],'end':segment["timestamp"][1]})
        
        def split_text_into_lines(data):
        
            MaxChars = 30
            #maxduration in seconds
            MaxDuration = 2.5
            #Split if nothing is spoken (gap) for these many seconds
            MaxGap = 1.5
        
            subtitles = []
            line = []
            line_duration = 0
            line_chars = 0
        
        
            for idx,word_data in enumerate(data):
                
                word = word_data["word"]
                start = word_data["start"]
                end = word_data["end"]
        
                line.append(word_data)
                line_duration += end - start
        
                temp = " ".join(item["word"] for item in line)
        
        
                # Check if adding a new word exceeds the maximum character count or duration
                new_line_chars = len(temp)
        
                duration_exceeded = line_duration > MaxDuration
                chars_exceeded = new_line_chars > MaxChars
                if idx>0:
                    gap = word_data['start'] - data[idx-1]['end']
                    maxgap_exceeded = gap > MaxGap
                else:
                    maxgap_exceeded = False
        
        
                if duration_exceeded or chars_exceeded or maxgap_exceeded:
                    if line:
                        subtitle_line = {
                            "word": " ".join(item["word"] for item in line),
                            "start": line[0]["start"],
                            "end": line[-1]["end"],
                            "textcontents": line
                        }
                        subtitles.append(subtitle_line)
                        line = []
                        line_duration = 0
                        line_chars = 0
        
        
            if line:
                subtitle_line = {
                    "word": " ".join(item["word"] for item in line),
                    "start": line[0]["start"],
                    "end": line[-1]["end"],
                    "textcontents": line
                }
                subtitles.append(subtitle_line)
        
            return subtitles
        
        linelevel_subtitles = split_text_into_lines(wordlevel_info)
        
        def create_caption(textJSON, framesize,font = "TheBoldFont-Bold",color='white', highlight_color='yellow',stroke_color='black',stroke_width=3.5):
            wordcount = len(textJSON['textcontents'])
            full_duration = textJSON['end']-textJSON['start']
        
            word_clips = []
            xy_textclips_positions =[]
        
            x_pos = 0
            y_pos = 0
            line_width = 0  # Total width of words in the current line
            frame_width = framesize[0]
            frame_height = framesize[1]
        
            x_buffer = frame_width*1/10
        
            max_line_width = frame_width - 2 * (x_buffer) + 2

            fontsize = int(frame_height * 0.050) #5 percent of video height
        
            space_width = ""
            space_height = ""

            highlight_word_index = 0
        
            for wordJSON in textJSON['textcontents']:
                duration = wordJSON['end']-wordJSON['start']
                word_clip = TextClip(wordJSON['word'].upper(), font = font,fontsize=fontsize,
                                      color=color,stroke_color=stroke_color,stroke_width=stroke_width).set_start(textJSON['start']).set_duration(full_duration)
                word_clip_space = TextClip(" ", font = font,fontsize=fontsize, color=color).set_start(textJSON['start']).set_duration(full_duration)
                word_width, word_height = word_clip.size
                space_width,space_height = word_clip_space.size
                
                if line_width + word_width+ space_width <= max_line_width:
                        # Store info of each word_clip created
                        xy_textclips_positions.append({
                            "x_pos":x_pos,
                            "y_pos": y_pos,
                            "width" : word_width,
                            "height" : word_height,
                            "word": wordJSON['word'],
                            "start": wordJSON['start'],
                            "end": wordJSON['end'],
                            "duration": duration
                        })
            
                        word_clip = word_clip.set_position((x_pos, y_pos))
                        word_clip_space = word_clip_space.set_position((x_pos+ word_width, y_pos))
            
                        x_pos = x_pos + word_width+ space_width
                        line_width = line_width+ word_width + space_width
                else:
                        # Move to the next line
                        x_pos = 0
                        y_pos = y_pos+ word_height+10
                        line_width = word_width + space_width
            
                        # Store info of each word_clip created
                        xy_textclips_positions.append({
                            "x_pos":x_pos,
                            "y_pos": y_pos,
                            "width" : word_width,
                            "height" : word_height,
                            "word": wordJSON['word'],
                            "start": wordJSON['start'],
                            "end": wordJSON['end'],
                            "duration": duration
                        })
            
                        word_clip = word_clip.set_position((x_pos, y_pos))
                        word_clip_space = word_clip_space.set_position((x_pos+ word_width , y_pos))
                        x_pos = word_width + space_width
        
        
                word_clips.append(word_clip)
                word_clips.append(word_clip_space)
        
        
            for highlight_word in xy_textclips_positions:

                highlight_word_index += 1
                if(highlight_word_index >= (len(xy_textclips_positions) - 1)):
                    word_clip_highlight = TextClip(highlight_word['word'].upper(), font = font,fontsize=fontsize, color="lime",
                                                   stroke_color=stroke_color,stroke_width=stroke_width).set_start(highlight_word['start']).set_duration(highlight_word['duration'])
                    word_clip_highlight = word_clip_highlight.set_position((highlight_word['x_pos'], highlight_word['y_pos']))
                    word_clips.append(word_clip_highlight)
                else:
                    word_clip_highlight = TextClip(highlight_word['word'].upper(), font = font,fontsize=fontsize, color=highlight_color,
                                                   stroke_color=stroke_color,stroke_width=stroke_width).set_start(highlight_word['start']).set_duration(highlight_word['duration'])
                    word_clip_highlight = word_clip_highlight.set_position((highlight_word['x_pos'], highlight_word['y_pos']))
                    word_clips.append(word_clip_highlight)
                
        
            return word_clips,xy_textclips_positions
        
        input_video = VideoFileClip(videofilename)
        frame_size = input_video.size
        
        all_linelevel_splits=[]
        
        for line in linelevel_subtitles:
            out_clips,positions = create_caption(line,frame_size)
        
            max_width = 0
            max_height = 0
        
            for position in positions:
                x_pos, y_pos = position['x_pos'],position['y_pos']
                width, height = position['width'],position['height']
            
                max_width = max(max_width, x_pos + width)

                
                max_height = y_pos + height + 100
        
            color_clip = ColorClip(size=(int(max_width*1.1), int(max_height*1.1)),
                                color=(64, 64, 64))
            color_clip = color_clip.set_opacity(.0)
            color_clip = color_clip.set_start(line['start']).set_duration(line['end']-line['start'])
            
            
            clip_to_overlay = CompositeVideoClip([color_clip]+ out_clips)

                
            clip_to_overlay = clip_to_overlay.set_position("bottom")
            
            
            all_linelevel_splits.append(clip_to_overlay)
        
        input_video_duration = input_video.duration
        
        
        final_video = CompositeVideoClip([input_video] + all_linelevel_splits)
        
        # Set the audio of the final video to be the same as the input video
        final_video = final_video.set_audio(input_video.audio)
        
        # Save the final clip as a video file with the audio included
        final_video.write_videofile(f"{output_folder_subs}/subtitled_{video}", fps=30, audio_bitrate="256k", codec='h264_nvenc', ffmpeg_params=["-ar", "48000"])

if __name__ == "__main__":
    main()
