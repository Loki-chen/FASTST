import numpy as np
import os
import threading


def load_data(file_path: str) -> np.ndarray:
    data = []
    with open(file_path, "r") as file:
        for i in file.readlines():
            buffer = i[:-1]
            if " " in buffer:
                data.append(list(map(int, buffer.split(" "))))
            elif  "," in buffer:
                data.append(list(map(int, buffer.split(","))))
            else:
                data.append([int(buffer)])
    return np.array(data)


def save_data(data: np.ndarray, file_path: str) -> None:
    buffer = ""
    for i in data:
        if i.shape[0] == 1:
            buffer += str(i[0])
        else:
            for index in range(i.shape[0]):
                if index == 0:
                    buffer += str(i[index])
                else: buffer += ("," + str(i[index]))
        buffer += "\n";
    with open(file_path, "w") as file:
        file.write(buffer)


def allocate_list(lst, n):
    if n == 1:
        return np.array(lst[start:end])
    size = len(lst) // n
    remainder = len(lst) % n
    result = []
    start = 0
    for i in range(n):
        end = start + size
        if i < remainder:
            end += 1
        result.append(np.array(lst[start:end]))
        start = end
    return result


def split_data(files: list, dir_path: str, alice_dir: str, bob_dir: str):
    for f in files:
        data = load_data(dir_path + f)
        mean, std = data.mean(), data.std()
        alice_data = np.random.normal(mean, std, data.shape)
        alice_data = np.round(alice_data).astype(int)
        bob_data = data - alice_data
        save_data(alice_data, alice_dir + f)
        save_data(bob_data, bob_dir + f)
        print(f"file {f} has splited")


def split_file(dir_path: str, alice_dir: str, bob_dir: str, n_threads = 1) -> None:
    if not os.path.exists(alice_dir):
        os.mkdir(alice_dir)
    if not os.path.exists(bob_dir):
        os.mkdir(bob_dir)

    files = os.listdir(dir_path)
    files_allocated = allocate_list(files, n_threads)
    threads = []
    for _ in range(n_threads):
        t = threading.Thread(target=split_data, args=(files_allocated[_], dir_path, alice_dir, bob_dir))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()


if __name__ == "__main__":
    dir_path = "/home/FASTST/data/bolt/quantize/mrpc/weights_txt/"
    alice_dir = "/home/FASTST/data/bolt/quantize/mrpc/alice_weights_txt/"
    bob_dir = "/home/FASTST/data/bolt/quantize/mrpc/bob_weights_txt/"
    try:
        split_file(dir_path, alice_dir, bob_dir, 8)
    except Exception as e:
        print(e)
        
        
