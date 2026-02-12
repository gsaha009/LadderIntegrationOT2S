from PIL import Image

imgs = [Image.open(f'/Users/gsaha/Downloads/5sigma_v2/nClusters_per_sensor_OG_{i}.png') for i in range(12)]

imgs[0].save(
    "/Users/gsaha/Downloads/5sigma_v2/output_5sigma_v2.gif",
    save_all=True,
    append_images=imgs[1:],
    duration=700,
    loop=0
)
