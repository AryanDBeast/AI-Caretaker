import speech_recognition as sr

print("\nAvailable microphones:")
for i, name in enumerate(sr.Microphone.list_microphone_names()):
    print(f"{i}: {name}")
print()