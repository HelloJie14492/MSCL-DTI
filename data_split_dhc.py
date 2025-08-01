import numpy as np

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

def split_dataset(dataset, ratio):
    n = int(ratio * len(dataset))
    dataset_1, dataset_2 = dataset[:n], dataset[n:]
    return dataset_1, dataset_2

def split(dir_input, input_file):
    sequences = np.load(dir_input + 'sequences.npy',allow_pickle=True)
    smiles = np.load(dir_input + 'smiles.npy', allow_pickle=True)
    molecule_words = np.load(dir_input + 'molecule_words.npy',allow_pickle=True)
    molecule_atoms = np.load(dir_input + 'molecule_atoms.npy',allow_pickle=True)
    molecule_adjs = np.load(dir_input + 'molecule_adjs.npy',allow_pickle=True)
    proteins = np.load(dir_input + 'proteins.npy',allow_pickle=True)
    interactions = np.load(dir_input + 'interactions.npy',allow_pickle=True)

    dataset = list(zip(molecule_words, molecule_atoms, molecule_adjs, proteins, interactions, sequences, smiles))
    dataset = shuffle_dataset(dataset, 1234)
    dataset_train, dataset_test = split_dataset(dataset, 0.833)

    molecule_words_train, molecule_atoms_train, molecule_adjs_train, proteins_train, interactions_train = [], [], [], [], []
    molecule_words_test, molecule_atoms_test, molecule_adjs_test, proteins_test, interactions_test = [], [], [], [], []
    sequences_train, smiles_train = [], []
    sequences_test, smiles_test = [], []


    for j in range(len(dataset_train)):
        molecule_word, molecule_atom, molecule_adj, protein, interaction, sequence, smile = dataset_train[j]
        molecule_words_train.append(molecule_word)
        molecule_atoms_train.append(molecule_atom)
        molecule_adjs_train.append(molecule_adj)
        proteins_train.append(protein)
        interactions_train.append(interaction)
        sequences_train.append(sequence)
        smiles_train.append(smile)

    for j in range(len(dataset_test)):
        molecule_word, molecule_atom, molecule_adj, protein, interaction, sequence, smile = dataset_test[j]
        molecule_words_test.append(molecule_word)
        molecule_atoms_test.append(molecule_atom)
        molecule_adjs_test.append(molecule_adj)
        proteins_test.append(protein)
        interactions_test.append(interaction)
        sequences_test.append(sequence)
        smiles_test.append(smile)

    molecule_words_train = np.asarray(molecule_words_train, dtype=object)
    molecule_atoms_train = np.asarray(molecule_atoms_train, dtype=object)
    molecule_adjs_train = np.asarray(molecule_adjs_train, dtype=object)
    proteins_train = np.asarray(proteins_train, dtype=object)
    interactions_train = np.asarray(interactions_train, dtype=object)
    sequences_train = np.asarray(sequences_train, dtype=object)
    smiles_train = np.asarray(smiles_train, dtype=object)
    np.save(input_file + '/train/molecule_words', molecule_words_train)
    np.save(input_file + '/train/molecule_atoms', molecule_atoms_train)
    np.save(input_file + '/train/molecule_adjs', molecule_adjs_train)
    np.save(input_file + '/train/proteins', proteins_train)
    np.save(input_file + '/train/interactions', interactions_train)
    np.save(input_file + '/train/sequences', sequences_train)
    np.save(input_file + '/train/smiles', smiles_train)

    molecule_words_test = np.asarray(molecule_words_test, dtype=object)
    molecule_atoms_test = np.asarray(molecule_atoms_test, dtype=object)
    molecule_adjs_test = np.asarray(molecule_adjs_test, dtype=object)
    proteins_test = np.asarray(proteins_test, dtype=object)
    interactions_test = np.asarray(interactions_test, dtype=object)
    sequences_test = np.asarray(sequences_test, dtype=object)
    smiles_test = np.asarray(smiles_test, dtype=object)
    np.save(input_file + '/test/molecule_words', molecule_words_test)
    np.save(input_file + '/test/molecule_atoms', molecule_atoms_test)
    np.save(input_file + '/test/molecule_adjs', molecule_adjs_test)
    np.save(input_file + '/test/proteins', proteins_test)
    np.save(input_file + '/test/interactions', interactions_test)
    np.save(input_file + '/test/sequences', sequences_test)
    np.save(input_file + '/test/smiles', smiles_test)

# split("/stu-3031/MMDG_Data/datasets/datasets/DrugBank/data_split/", "/stu-3031/MMDG_Data/datasets/datasets/DrugBank")
# split("/stu-3031/MMDG_Data/datasets/datasets/Human/data_split/", "/stu-3031/MMDG_Data/datasets/datasets/Human")
split("/stu-3031/MMDG_Data/datasets/datasets/C.elegans/data_split/", "/stu-3031/MMDG_Data/datasets/datasets/C.elegans")
# split("Davis/data_split/", "Davis")
# split("Kiba/data_split/", "Kiba")
