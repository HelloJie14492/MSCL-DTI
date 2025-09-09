import json
import numpy as np
import os
DATA_ROOT = os.getenv("DATA_ROOT", "/stu-3031/MMDG_Data/datasets/datasets/")
# data_fold = json.load(open("davis_div.txt"))
dir_input = "/stu-3031/MMDG_Data/datasets/datasets/Davis/data_split/"
sequences = np.load(dir_input + 'sequences.npy',allow_pickle=True)
smiles = np.load(dir_input + 'smiles.npy', allow_pickle=True)
molecule_words = np.load(dir_input + 'molecule_words.npy',allow_pickle=True)
molecule_atoms = np.load(dir_input + 'molecule_atoms.npy',allow_pickle=True)
molecule_adjs = np.load(dir_input + 'molecule_adjs.npy',allow_pickle=True)
proteins = np.load(dir_input + 'proteins.npy',allow_pickle=True)
affinity = np.load(dir_input + 'interactions.npy',allow_pickle=True)
print(len(smiles))

with open(dir_input + "train.txt", "r") as f:  # 打开文件
    train_list = f.read()
    train_list = train_list.split(', ')
    train_list = np.array(train_list,dtype=int)
    # train_list = np.array(train_list, dtype=object)
print(len(train_list))
with open(dir_input + "test.txt", "r") as f:  # 打开文件
    test_list = f.read()
    test_list = test_list.split(', ')
    test_list = np.array(test_list,dtype=int)
    # test_list = np.array(test_list, dtype=object)

# train_fold=np.loadtxt(dir_input + 'train.txt', dtype=np.int)
print(len(test_list))
molecule_words_train, molecule_atoms_train, molecule_adjs_train, proteins_train, affinity_train = [], [], [], [], []
molecule_words_test, molecule_atoms_test, molecule_adjs_test, proteins_test, affinity_test = [], [], [], [], []
sequences_train, smiles_train = [], []
sequences_test, smiles_test = [], []

a = 0
for j in range(len(train_list)):
    if train_list[j] >= 25772: continue
    a += 1
    molecule_words_train.append(np.array(molecule_words[train_list[j]]))
    molecule_atoms_train.append(molecule_atoms[train_list[j]])
    molecule_adjs_train.append(molecule_adjs[train_list[j]])
    proteins_train.append(proteins[train_list[j]])
    affinity_train.append(affinity[train_list[j]])
    sequences_train.append(sequences[train_list[j]])
    smiles_train.append(smiles[train_list[j]])
b = 0
for j in range(len(test_list)):
    if test_list[j] >= 25772: continue
    b += 1
    molecule_words_test.append(np.array(molecule_words[test_list[j]]))
    molecule_atoms_test.append(molecule_atoms[test_list[j]])
    molecule_adjs_test.append(molecule_adjs[test_list[j]])
    proteins_test.append(proteins[test_list[j]])
    affinity_test.append(affinity[test_list[j]])
    sequences_test.append(sequences[test_list[j]])
    smiles_test.append(smiles[test_list[j]])

print(a, b)

molecule_words_train = np.asarray(molecule_words_train, dtype=object)
molecule_atoms_train = np.asarray(molecule_atoms_train, dtype=object)
molecule_adjs_train = np.asarray(molecule_adjs_train, dtype=object)
proteins_train = np.asarray(proteins_train, dtype=object)
affinity_train = np.asarray(affinity_train, dtype=object)
sequences_train = np.asarray(sequences_train, dtype=object)
smiles_train = np.asarray(smiles_train, dtype=object)
np.save(os.path.join(DATA_ROOT, "Davis", "train", "molecule_words.npy"), molecule_words_train)
np.save(os.path.join(DATA_ROOT, "Davis", "train", "molecule_atoms.npy"), molecule_atoms_train)
np.save(os.path.join(DATA_ROOT, "Davis", "train", "molecule_adjs.npy"), molecule_adjs_train)
np.save(os.path.join(DATA_ROOT, "Davis", "train", "proteins.npy"), proteins_train)
np.save(os.path.join(DATA_ROOT, "Davis", "train", "interactions.npy"), affinity_train)
np.save(os.path.join(DATA_ROOT, "Davis", "train", "sequences.npy"), sequences_train)
np.save(os.path.join(DATA_ROOT, "Davis", "train", "smiles.npy"), smiles_train)

molecule_words_test = np.asarray(molecule_words_test, dtype=object)
molecule_atoms_test = np.asarray(molecule_atoms_test, dtype=object)
molecule_adjs_test = np.asarray(molecule_adjs_test, dtype=object)
proteins_test = np.asarray(proteins_test, dtype=object)
affinity_test = np.asarray(affinity_test, dtype=object)
sequences_test = np.asarray(sequences_test, dtype=object)
smiles_test = np.asarray(smiles_test, dtype=object)
np.save(os.path.join(DATA_ROOT, "Davis", "test", "molecule_words.npy"), molecule_words_test)
np.save(os.path.join(DATA_ROOT, "Davis", "test", "molecule_atoms.npy"), molecule_atoms_test)
np.save(os.path.join(DATA_ROOT, "Davis", "test", "molecule_adjs.npy"), molecule_adjs_test)
np.save(os.path.join(DATA_ROOT, "Davis", "test", "proteins.npy"), proteins_test)
np.save(os.path.join(DATA_ROOT, "Davis", "test", "interactions.npy"), affinity_test)
np.save(os.path.join(DATA_ROOT, "Davis", "test", "sequences.npy"), sequences_test)
np.save(os.path.join(DATA_ROOT, "Davis", "test", "smiles.npy"), smiles_test)