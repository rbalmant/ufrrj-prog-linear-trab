from zipfile import ZipFile

def unzip_files_into_memory(filename):
    input_file = ZipFile(filename)
    content = {name: input_file.read(name) for name in input_file.namelist()}
    return content
