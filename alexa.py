
# Imports
import pyaudio
import numpy as np
from openwakeword.model import Model
import argparse
import wave
import matplotlib.pyplot as plt

# Parse input arguments
parser=argparse.ArgumentParser()
parser.add_argument(
    "--chunk_size",
    help="How much audio (in number of samples) to predict on at once",
    type=int,
    default=1280,
    required=False
)
parser.add_argument(
    "--model_path",
    help="The path of a specific model to load",
    type=str,
    default="",
    required=False
)
parser.add_argument(
    "--inference_framework",
    help="The inference framework to use (either 'onnx' or 'tflite'",
    type=str,
    default='tflite',
    required=False
)

args=parser.parse_args()

# Get microphone stream
# FORMAT = pyaudio.paInt16
# CHANNELS = 1
# RATE = 16000
# CHUNK = args.chunk_size
# audio = pyaudio.PyAudio()
# mic_stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

# Load pre-trained openwakeword models
if args.model_path != "":
    owwModel = Model(wakeword_models=[args.model_path], inference_framework=args.inference_framework)
else:
    owwModel = Model(inference_framework=args.inference_framework)

n_models = len(owwModel.models.keys())

# Open the audio file
wav_file = wave.open('bagga sample 1.wav', 'rb')
CHANNELS = wav_file.getnchannels()
RATE = wav_file.getframerate()
print(RATE)
CHUNK = args.chunk_size

audio = pyaudio.PyAudio()

# MASKING 
duration = 39.4 #seconds
sample_rate = 44100
chunk_size = 1280

total_samples = int(duration * sample_rate)
num_chunks = total_samples // chunk_size

print(num_chunks)  # 137

def timestamp_to_chunk(timestamp, sample_rate=44100, chunk_size=1280):
    samples_per_chunk = chunk_size
    chunk_number = int(timestamp * sample_rate / samples_per_chunk)
    return chunk_number


print(timestamp_to_chunk(2.75)) # chunks saying alexa from 67 to 94

ground_truth = [0 for i in range(1357)]
for i in range(150):
    if i>=67 and i<=94:
        ground_truth[i] += 1

print(ground_truth)

# Open stream
stream = audio.open(format=audio.get_format_from_width(wav_file.getsampwidth()),
                    channels=CHANNELS,
                    rate=RATE,
                    output=True)

# # Run capture loop continuosly, checking for wakewords
# if __name__ == "__main__":
#     # Generate output string header
#     print("\n\n")
#     print("#"*100)
#     print("Listening for wakewords...")
#     print("#"*100)
#     print("\n"*(n_models*3))

#     predictions = []

#     try:
#         while True:
#             # Get audio
#             raw_audio = wav_file.readframes(CHUNK)
#             if not raw_audio:
#                 break 
#             audio = np.frombuffer(raw_audio, dtype=np.int16)

#             # Feed to openWakeWord model
#             prediction = owwModel.predict(audio)

#             predictions.append(prediction['alexa'])

#             stream.write(raw_audio)
            
#             # Column titles
#             n_spaces = 16
#             output_string_header = """
#                 Model Name         | Score | Wakeword Status
#                 --------------------------------------
#                 """

#             for mdl in owwModel.prediction_buffer.keys():
#                 # Add scores in formatted table
#                 scores = list(owwModel.prediction_buffer[mdl])
#                 curr_score = format(scores[-1], '.20f').replace("-", "")

#                 output_string_header += f"""{mdl}{" "*(n_spaces - len(mdl))}   | {curr_score[0:5]} | {"--"+" "*20 if scores[-1] <= 0.5 else "Wakeword Detected!"}
#                 """

#             # Print results table
#             print("\033[F"*(4*n_models+1))
#             print(output_string_header, "                             ", end='\r')
        
#         print('total predictions done:' ,len(predictions))
#         print(predictions)

#         threshold = 0.001

#         binary_predictions = [1 if pred >= threshold else 0 for pred in predictions]
#         print(binary_predictions)

#         tp = sum(gt == 1 and pred == 1 for gt, pred in zip(ground_truth, binary_predictions))
#         fp = sum(gt == 0 and pred == 1 for gt, pred in zip(ground_truth, binary_predictions))
#         tn = sum(gt == 0 and pred == 0 for gt, pred in zip(ground_truth, binary_predictions))
#         fn = sum(gt == 1 and pred == 0 for gt, pred in zip(ground_truth, binary_predictions))

#         precision = tp / (tp + fp) if (tp + fp) > 0 else 0
#         recall = tp / (tp + fn) if (tp + fn) > 0 else 0
#         accuracy = (tp + tn) / (tp + fp + tn + fn) if (tp + fp + tn + fn) > 0 else 0

#         print(f"True Positives (TP): {tp}")
#         print(f"False Positives (FP): {fp}")
#         print(f"True Negatives (TN): {tn}")
#         print(f"False Negatives (FN): {fn}")
#         print(f"Precision: {precision:.2f}")
#         print(f"Recall: {recall:.2f}")
#         print(f"Accuracy: {accuracy * 100:.2f}%")

#         plt.figure(figsize=(10, 6))
#         plt.plot(binary_predictions, label='Predictions', color='blue')
#         plt.plot(ground_truth, label='Ground Truth', linestyle='--', color='green')
#         plt.xlabel('Chunk Number')
#         plt.ylabel('Value (0 or 1)')
#         plt.title('Predictions vs Ground Truth')
#         plt.legend()
#         plt.show()
#     except KeyboardInterrupt:
#         print("\nUser interrupted.")
        
#     finally:
#         # Stop ss.close()
#         stream.stop_stream()
#         stream.close()
#         # audio.terminate()
#         wav_file.close()
