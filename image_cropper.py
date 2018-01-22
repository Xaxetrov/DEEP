def crop(save_file, image, position, scale, strides, base_chunk_size=36):
    box = (position[0] * strides[0] / scale,
           position[1] * strides[1] / scale,
           (position[0] * strides[0] + base_chunk_size) / scale,
           (position[1] * strides[1] + base_chunk_size) / scale)
    a = image.crop(box)
    a = a.resize((base_chunk_size, base_chunk_size))
    a.save(save_file)
