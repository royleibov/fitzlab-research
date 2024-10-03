from PIL import Image

# Open the existing GIF
gif = Image.open('fourier_transform_animation.gif')

# Create a list to hold each frame
frames = []

# Iterate through each frame in the original GIF
try:
    while True:
        # Copy the current frame and append it to the frames list
        frame = gif.copy()
        frames.append(frame)
        gif.seek(gif.tell() + 1)
except EOFError:
    pass

# Limit the frames to the first 7 frames
frames = frames[:7]

# Save the new GIF with a longer duration per frame (in milliseconds).
# Default is usually around 100ms per frame. To slow it down, increase the value (e.g., to 200ms).
frames[0].save('fourier_transform_animation_slow.gif', save_all=True, append_images=frames[1:], duration=700, loop=0)