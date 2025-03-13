import whisperx

def transcribe_audio(audio_file):
    device = "cpu" 
    batch_size = 16 # reduce if low on GPU mem
    compute_type = "int8" # change to "int8" if low on GPU mem (may reduce accuracy)

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, language="en" ,batch_size=batch_size)

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model_a

    # 3. Assign speaker labels
    diarize_model = whisperx.DiarizationPipeline(use_auth_token="", device=device)

    # add min/max number of speakers if known
    diarize_segments = diarize_model(audio)

    result = whisperx.assign_word_speakers(diarize_segments, result)

    combined_text = " ".join(item['text'].strip() for item in result["segments"])
    return(combined_text)

