import numpy as np
from rdkit import Chem
from rdkit.Chem.rdchem import BondType
from graph_features import atom_features
from collections import defaultdict
from transformers import AutoModel, AutoTokenizer
import torch
import pickle

BONDTYPE_TO_INT = defaultdict(
    lambda: 0,
    {
        BondType.SINGLE: 0,
        BondType.DOUBLE: 1,
        BondType.TRIPLE: 2,
        BondType.AROMATIC: 3
    }
)


protein_dict = { "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
                 "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
                 "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
                 "U": 19, "T": 20, "W": 21,
                 "V": 22, "Y": 23, "X": 24,
                 "Z": 25 }

CHAR_SMI_SET = {"(": 1, ".": 2, "0": 3, "2": 4, "4": 5, "6": 6, "8": 7, "@": 8,
                "B": 9, "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
                "T": 17, "V": 18, "Z": 19, "\\": 20, "b": 21, "d": 22, "f": 23, "h": 24,
                "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
                "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
                "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
                "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
                "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64}

num_atom_feat = 34

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return [x == s for s in allowable_set]

def smile_to_graph(smile):
    molecule = Chem.MolFromSmiles(smile)
    n_atoms = molecule.GetNumAtoms()
    atoms = [molecule.GetAtomWithIdx(i) for i in range(n_atoms)]

    adjacency = Chem.rdmolops.GetAdjacencyMatrix(molecule)
    node_features = np.array([atom_features(atom) for atom in atoms])

    n_edge_features = 4
    edge_features = np.zeros([n_atoms, n_atoms, n_edge_features])
    for bond in molecule.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        bond_type = BONDTYPE_TO_INT[bond.GetBondType()]
        edge_features[i, j, bond_type] = 1
        edge_features[j, i, bond_type] = 1

    return node_features, adjacency

def first_sequence(sequence):
    words = [protein_dict[sequence[i]]
             for i in range(len(sequence))]
    return np.array(words)


import os

# 指定使用 GPU 4
os.environ['CUDA_VISIBLE_DEVICES'] = '6'  # 只让 PyTorch 看见 GPU 4

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('The code uses GPU')
else:
    device = torch.device('cpu')
    print('The code uses CPU!!!')

# #
# prot_tokenizer = AutoTokenizer.from_pretrained("./models/prot_bert_bfd", do_lower_case=False)  #  Rostlab
# prot_model = AutoModel.from_pretrained("./models/prot_bert_bfd").to(device)


# 替换为 esm2_t36_3B_UR50D
prot_tokenizer = AutoTokenizer.from_pretrained("/stu-3031/MMDG-Revise/models/esm2_t36_3B_UR50D", do_lower_case=False)
prot_model = AutoModel.from_pretrained("/stu-3031/MMDG-Revise/models/esm2_t36_3B_UR50D").to(device)
# import esm
#
# # 加载模型
# prot_model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
# prot_model = prot_model.to(device)
# batch_converter = alphabet.get_batch_converter()

# # 在文件顶部添加导入
# from transformers import EsmTokenizer, EsmModel
#
# # 替换原模型加载代码
# prot_tokenizer = EsmTokenizer.from_pretrained("./models/esm2_t36_3B_UR50D")
# prot_model = EsmModel.from_pretrained("./models/esm2_t36_3B_UR50D").to(device)


# chem_tokenizer = AutoTokenizer.from_pretrained("./models/PubChem10M_SMILES_BPE_450k", do_lower_case=False)   #  seyonec
# chem_model = AutoModel.from_pretrained("./models/PubChem10M_SMILES_BPE_450k").to(device)

# 修改后的代码
chem_tokenizer = AutoTokenizer.from_pretrained(
    "/stu-3031/Model/MoLFormer-XL-both-10pct",
    trust_remote_code=True  # 添加这个关键参数
)
chem_model = AutoModel.from_pretrained(
    "/stu-3031/Model/MoLFormer-XL-both-10pct",
    trust_remote_code=True  # 添加这个关键参数
).to(device)


def DTI_datasets(dataset, dir_input):  # BindingDB, Human, C.elegan, GPCRs
    with open(dataset,"r") as f:
        data_list = f.read().strip().split('\n')
    """Exclude data contains '.' in the SMILES format."""
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    N = len(data_list)
    molecule_atoms, molecule_adjs, molecule_words, proteins, sequencess, smiless, interactions = [], [], [], [], [], [], []
    p_LM, d_LM = {}, {}
    for no, data in enumerate(data_list):
        print('/'.join(map(str, [no + 1, N])))

        smiles, sequences, interaction = data.strip().split(" ")
        # if len(sequences) > 5000:
        #     sequences = sequences[0:5000]
        # 修改为（截断到 1024）
        if len(sequences) > 1024:
            sequences = sequences[0:1024]
        sequencess.append(sequences)
        smiless.append(smiles)
        # print(len(sequences))
        # protein_input = prot_tokenizer.batch_encode_plus([" ".join(sequences)], add_special_tokens=True, padding=True)#"longest", max_length=1200, truncation=True, return_tensors='pt')
        # p_IDS = torch.tensor(protein_input["input_ids"]).to(device)
        # p_a_m = torch.tensor(protein_input["attention_mask"]).to(device)
        # # sequences = torch.tensor(sequences).to(self.device)
        # with torch.no_grad():
        #     prot_outputs = prot_model(input_ids=p_IDS, attention_mask=p_a_m)
        # prot_feature = prot_outputs.last_hidden_state.squeeze(0).to('cpu').data.numpy()

        # 修改编码方式（移除空格）
        protein_input = prot_tokenizer.batch_encode_plus(
            [sequences],  # 直接传入序列，不加空格
            add_special_tokens=True,
            padding=True,
            return_tensors='pt'
        )

        # 添加设备移动代码
        protein_input = {k: v.to(device) for k, v in protein_input.items()}

        with torch.no_grad():
            prot_outputs = prot_model(**protein_input)
        prot_feature = prot_outputs.last_hidden_state.squeeze(0).to('cpu').data.numpy()

        if sequences not in p_LM:
            p_LM[sequences] = prot_feature

        # chem_input = chem_tokenizer.batch_encode_plus([smiles], add_special_tokens=True, padding=True)
        # c_IDS = torch.tensor(chem_input["input_ids"]).to(device)
        # c_a_m = torch.tensor(chem_input["attention_mask"]).to(device)
        # 修改后的特征提取代码
        chem_input = chem_tokenizer(
            smiles,
            add_special_tokens=True,
            padding="max_length",  # 确保填充到最大长度
            max_length=512,  # 设置最大长度
            truncation=True,  # 启用截断
            return_tensors="pt"
        )

        # 修改后的调用代码
        c_IDS = chem_input["input_ids"].to(device)
        c_a_m = chem_input["attention_mask"].to(device)

        with torch.no_grad():
            chem_outputs = chem_model(input_ids=c_IDS, attention_mask=c_a_m)
        chem_feature = chem_outputs.last_hidden_state.squeeze(0).to('cpu').data.numpy()  # .mean(dim=1)
        if smiles not in d_LM:
            d_LM[smiles] = chem_feature

        molecule_word = []
        for i in range(len(smiles)):
            molecule_word.append(CHAR_SMI_SET[smiles[i]])
        molecule_word = np.array(molecule_word)
        molecule_words.append(molecule_word)

        atom_feature, adj = smile_to_graph(smiles)
        molecule_atoms.append(atom_feature)
        molecule_adjs.append(adj)

        protein_first = first_sequence(sequences)
        proteins.append(protein_first)

        interactions.append(np.array([float(interaction)]))

    with open(dir_input + "p_LM.pkl", "wb") as p:
        pickle.dump(p_LM, p)

    with open(dir_input + "d_LM.pkl", "wb") as d:
        pickle.dump(d_LM, d)

    # molecule_atoms：分子原子特征（来自smile_to_graph）。
    # molecule_adjs：分子邻接矩阵（化学键信息）。
    # molecule_words：SMILES字符级编码（CHAR_SMI_SET映射）。
    # proteins：蛋白质序列的初级编码（protein_dict映射）。
    # sequencess：原始蛋白质序列。
    # smiless：原始SMILES字符串。
    # interactions：相互作用标签（0 / 1或浮点数）。
    # p_LM、d_LM：蛋白质和分子的语言模型特征缓存（避免重复计算）。
    molecule_words = np.asarray(molecule_words, dtype=object)
    molecule_atoms = np.asarray(molecule_atoms, dtype=object)
    molecule_adjs = np.asarray(molecule_adjs, dtype=object)
    proteins = np.asarray(proteins, dtype=object)
    sequencess = np.asarray(sequencess, dtype=object)
    smiless = np.asarray(smiless, dtype=object)
    interactions = np.asarray(interactions, dtype=object)
    np.save(dir_input + 'molecule_words', molecule_words)
    np.save(dir_input + 'molecule_atoms', molecule_atoms)
    np.save(dir_input + 'molecule_adjs', molecule_adjs)
    np.save(dir_input + 'proteins', proteins)
    np.save(dir_input + 'sequences', sequencess)
    np.save(dir_input + 'smiles', smiless)
    np.save(dir_input + 'interactions', interactions)

def DTI_drugbank_datasets(dataset, dir_input):  # DrugBank
    with open(dataset,"r") as f:
        data_list = f.read().strip().split('\n')
    """Exclude data contains '.' in the SMILES format."""
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    N = len(data_list)
    molecule_atoms, molecule_adjs, molecule_words, proteins, sequencess, smiless, interactions = [], [], [], [], [], [], []
    p_LM, d_LM = {}, {}
    for no, data in enumerate(data_list):
        # print('/'.join(map(str, [no + 1, N])))
        if (no + 1) % 100 == 0 or (no + 1) == N:
            print('/'.join(map(str, [no + 1, N])))

        _, _, smiles, sequences, interaction = data.strip().split(" ")
        if len(sequences) > 1024:
            sequences = sequences[0:1024]
        # if len(smiles) > 512:
        #     smiles = smiles[0:512]
        sequencess.append(sequences)
        smiless.append(smiles)
        # print(len(sequences))
        # protein_input = prot_tokenizer.batch_encode_plus([" ".join(sequences)], add_special_tokens=True, padding=True)#"longest", max_length=1200, truncation=True, return_tensors='pt')
        # p_IDS = torch.tensor(protein_input["input_ids"]).to(device)
        # p_a_m = torch.tensor(protein_input["attention_mask"]).to(device)
        # # sequences = torch.tensor(sequences).to(self.device)
        # with torch.no_grad():
        #     prot_outputs = prot_model(input_ids=p_IDS, attention_mask=p_a_m)
        # prot_feature = prot_outputs.last_hidden_state.squeeze(0).to('cpu').data.numpy()

        # 修改编码方式（移除空格）
        protein_input = prot_tokenizer.batch_encode_plus(
            [sequences],  # 直接传入序列，不加空格
            add_special_tokens=True,
            padding=True,
            return_tensors='pt'
        )

        # 添加设备移动代码
        protein_input = {k: v.to(device) for k, v in protein_input.items()}

        with torch.no_grad():
            prot_outputs = prot_model(**protein_input)
        prot_feature = prot_outputs.last_hidden_state.squeeze(0).to('cpu').data.numpy()

        # # 在数据处理部分
        # batch = [(f"seq_{i}", sequence) for i, sequence in enumerate(sequences)]
        # batch_labels, batch_strs, batch_tokens = batch_converter(batch)
        # batch_tokens = batch_tokens.to(device)
        #
        # with torch.no_grad():
        #     results = prot_model(batch_tokens, repr_layers=[36])  # 36 是最后一层
        # prot_feature = results["representations"][36]
        if sequences not in p_LM:
            p_LM[sequences] = prot_feature
        # print(len(smiles))
        if len(smiles) < 512:
            # chem_input = chem_tokenizer.batch_encode_plus([smiles], add_special_tokens=True, padding=True)
            # 修改后的特征提取代码
            chem_input = chem_tokenizer(
                smiles,
                add_special_tokens=True,
                padding="max_length",  # 确保填充到最大长度
                max_length=512,  # 设置最大长度
                truncation=True,  # 启用截断
                return_tensors="pt"
            )


            # c_IDS = torch.tensor(chem_input["input_ids"]).to(device)
            # c_a_m = torch.tensor(chem_input["attention_mask"]).to(device)

            # 修改后的调用代码
            c_IDS = chem_input["input_ids"].to(device)
            c_a_m = chem_input["attention_mask"].to(device)
            with torch.no_grad():
                chem_outputs = chem_model(input_ids=c_IDS, attention_mask=c_a_m)
            chem_feature = chem_outputs.last_hidden_state.squeeze(0).to('cpu').data.numpy()  # .mean(dim=1)
            if smiles not in d_LM:
                d_LM[smiles] = chem_feature
        else:
            smiles_short = smiles[0:512]
            chem_input = chem_tokenizer.batch_encode_plus([smiles_short], add_special_tokens=True, padding=True)
            c_IDS = torch.tensor(chem_input["input_ids"]).to(device)
            c_a_m = torch.tensor(chem_input["attention_mask"]).to(device)
            with torch.no_grad():
                chem_outputs = chem_model(input_ids=c_IDS, attention_mask=c_a_m)
            chem_feature = chem_outputs.last_hidden_state.squeeze(0).to('cpu').data.numpy()  # .mean(dim=1)
            if smiles not in d_LM:
                d_LM[smiles] = chem_feature

        molecule_word = []
        for i in range(len(smiles)):
            molecule_word.append(CHAR_SMI_SET[smiles[i]])
        molecule_word = np.array(molecule_word)
        molecule_words.append(molecule_word)

        atom_feature, adj = smile_to_graph(smiles)
        molecule_atoms.append(atom_feature)
        molecule_adjs.append(adj)

        protein_first = first_sequence(sequences)
        proteins.append(protein_first)

        interactions.append(np.array([float(interaction)]))

    with open(dir_input + "p_LM.pkl", "wb") as p:
        pickle.dump(p_LM, p)

    with open(dir_input + "d_LM.pkl", "wb") as d:
        pickle.dump(d_LM, d)

    molecule_words = np.asarray(molecule_words, dtype=object)
    molecule_atoms = np.asarray(molecule_atoms, dtype=object)
    molecule_adjs = np.asarray(molecule_adjs, dtype=object)
    proteins = np.asarray(proteins, dtype=object)
    sequencess = np.asarray(sequencess, dtype=object)
    smiless = np.asarray(smiless, dtype=object)
    interactions = np.asarray(interactions, dtype=object)
    np.save(dir_input + 'molecule_words', molecule_words)
    np.save(dir_input + 'molecule_atoms', molecule_atoms)
    np.save(dir_input + 'molecule_adjs', molecule_adjs)
    np.save(dir_input + 'proteins', proteins)
    np.save(dir_input + 'sequences', sequencess)
    np.save(dir_input + 'smiles', smiless)
    np.save(dir_input + 'interactions', interactions)

def DTA_datasets(dataset, dir_input): # Davis, Kiba
    with open(dataset,"r") as f:
        data_list = f.read().strip().split('\n')
    """Exclude data contains '.' in the SMILES format."""
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    N = len(data_list)
    molecule_atoms, molecule_adjs, molecule_words, proteins, affinities = [], [], [], [], []
    p_LM, d_LM = {}, {}
    p_sequences, d_smiles = [], []
    for no, data in enumerate(data_list):
        print('/'.join(map(str, [no + 1, N])))

        _, _, smiles, sequences, affinity = data.strip().split(" ")
        p_sequences.append(sequences)
        d_smiles.append(smiles)
        protein_input = prot_tokenizer.batch_encode_plus([" ".join(sequences)], add_special_tokens=True,
                                                         padding=True)  # "longest", max_length=1200, truncation=True, return_tensors='pt')
        p_IDS = torch.tensor(protein_input["input_ids"]).to(device)
        p_a_m = torch.tensor(protein_input["attention_mask"]).to(device)
        # sequences = torch.tensor(sequences).to(self.device)
        with torch.no_grad():
            prot_outputs = prot_model(input_ids=p_IDS, attention_mask=p_a_m)
        prot_feature = prot_outputs.last_hidden_state.squeeze(0).to('cpu').data.numpy()
        if sequences not in p_LM:
            p_LM[sequences] = prot_feature
        print(len(smiles))
        if len(smiles) < 512:
            chem_input = chem_tokenizer.batch_encode_plus([smiles], add_special_tokens=True, padding=True)
            c_IDS = torch.tensor(chem_input["input_ids"]).to(device)
            c_a_m = torch.tensor(chem_input["attention_mask"]).to(device)
            with torch.no_grad():
                chem_outputs = chem_model(input_ids=c_IDS, attention_mask=c_a_m)
            chem_feature = chem_outputs.last_hidden_state.squeeze(0).to('cpu').data.numpy()  # .mean(dim=1)
            if smiles not in d_LM:
                d_LM[smiles] = chem_feature
        else:
            smiles_short = smiles[0:512]
            chem_input = chem_tokenizer.batch_encode_plus([smiles_short], add_special_tokens=True, padding=True)
            c_IDS = torch.tensor(chem_input["input_ids"]).to(device)
            c_a_m = torch.tensor(chem_input["attention_mask"]).to(device)
            with torch.no_grad():
                chem_outputs = chem_model(input_ids=c_IDS, attention_mask=c_a_m)
            chem_feature = chem_outputs.last_hidden_state.squeeze(0).to('cpu').data.numpy()  # .mean(dim=1)
            if smiles not in d_LM:
                d_LM[smiles] = chem_feature

        molecule_word = []
        for i in range(len(smiles)):
            molecule_word.append(CHAR_SMI_SET[smiles[i]])
        molecule_word = np.array(molecule_word)
        molecule_words.append(molecule_word)

        atom_feature, adj = smile_to_graph(smiles)
        molecule_atoms.append(atom_feature)
        molecule_adjs.append(adj)

        protein_first = first_sequence(sequences)
        proteins.append(protein_first)
        affinities.append(np.array([float(affinity)]))

    with open(dir_input + "p_LM.pkl", "wb") as p:
        pickle.dump(p_LM, p)

    with open(dir_input + "d_LM.pkl", "wb") as d:
        pickle.dump(d_LM, d)

    np.save(dir_input + 'sequences', p_sequences)
    np.save(dir_input + 'smiles', d_smiles)
    np.save(dir_input + 'molecule_words', molecule_words)
    np.save(dir_input + 'molecule_atoms', molecule_atoms)
    np.save(dir_input + 'molecule_adjs', molecule_adjs)
    np.save(dir_input + 'proteins', proteins)
    np.save(dir_input + 'affinity', affinities)

def DTI_datasets3(dataset, dir_input):  # BindingDB, Human, C.elegan, GPCRs
    with open(dataset,"r") as f:
        data_list = f.read().strip().split('\n')
    """Exclude data contains '.' in the SMILES format."""
    data_list = [d for d in data_list if '.' not in d.strip().split()[0]]
    N = len(data_list)
    molecule_atoms, molecule_adjs, molecule_words, proteins, sequencess, smiless, interactions = [], [], [], [], [], [], []
    # p_LM, d_LM = {}, {}
    for no, data in enumerate(data_list):
        print('/'.join(map(str, [no + 1, N])))

        smiles, sequences, interaction = data.strip().split(" ")
        if len(sequences) > 5000:
            sequences = sequences[0:5000]
        sequencess.append(sequences)
        smiless.append(smiles)
        # print(len(sequences))
        # protein_input = prot_tokenizer.batch_encode_plus([" ".join(sequences)], add_special_tokens=True, padding=True)#"longest", max_length=1200, truncation=True, return_tensors='pt')
        # p_IDS = torch.tensor(protein_input["input_ids"]).to(device)
        # p_a_m = torch.tensor(protein_input["attention_mask"]).to(device)
        # # sequences = torch.tensor(sequences).to(self.device)
        # with torch.no_grad():
        #     prot_outputs = prot_model(input_ids=p_IDS, attention_mask=p_a_m)
        # prot_feature = prot_outputs.last_hidden_state.squeeze(0).to('cpu').data.numpy()
        # if sequences not in p_LM:
        #     p_LM[sequences] = prot_feature

        # chem_input = chem_tokenizer.batch_encode_plus([smiles], add_special_tokens=True, padding=True)
        # c_IDS = torch.tensor(chem_input["input_ids"]).to(device)
        # c_a_m = torch.tensor(chem_input["attention_mask"]).to(device)
        # with torch.no_grad():
        #     chem_outputs = chem_model(input_ids=c_IDS, attention_mask=c_a_m)
        # chem_feature = chem_outputs.last_hidden_state.squeeze(0).to('cpu').data.numpy()  # .mean(dim=1)
        # if smiles not in d_LM:
        #     d_LM[smiles] = chem_feature

        molecule_word = []
        for i in range(len(smiles)):
            molecule_word.append(CHAR_SMI_SET[smiles[i]])
        molecule_word = np.array(molecule_word)
        molecule_words.append(molecule_word)

        atom_feature, adj = smile_to_graph(smiles)
        molecule_atoms.append(atom_feature)
        molecule_adjs.append(adj)

        protein_first = first_sequence(sequences)
        proteins.append(protein_first)

        interactions.append(np.array([float(interaction)]))

    # with open(dir_input + "p_LM.pkl", "wb") as p:
    #     pickle.dump(p_LM, p)
    #
    # with open(dir_input + "d_LM.pkl", "wb") as d:
    #     pickle.dump(d_LM, d)

    molecule_words = np.asarray(molecule_words, dtype=object)
    molecule_atoms = np.asarray(molecule_atoms, dtype=object)
    molecule_adjs = np.asarray(molecule_adjs, dtype=object)
    proteins = np.asarray(proteins, dtype=object)
    sequencess = np.asarray(sequencess, dtype=object)
    smiless = np.asarray(smiless, dtype=object)
    interactions = np.asarray(interactions, dtype=object)
    np.save(dir_input + 'molecule_words', molecule_words)
    np.save(dir_input + 'molecule_atoms', molecule_atoms)
    np.save(dir_input + 'molecule_adjs', molecule_adjs)
    np.save(dir_input + 'proteins', proteins)
    np.save(dir_input + 'sequences', sequencess)
    np.save(dir_input + 'smiles', smiless)
    np.save(dir_input + 'interactions', interactions)


if __name__ == "__main__":

    # DTI_datasets3("BindingDB_cold/unseen_pair_setting/Unseen pair/train.txt", 'BindingDB_cold/unseen_pair_setting/train/')
    # DTI_datasets3("BindingDB_cold/unseen_pair_setting/Unseen pair/test.txt", 'BindingDB_cold/unseen_pair_setting/test/')
    # DTI_datasets3("BindingDB_cold/unseen_pair_setting/Unseen pair/dev.txt", 'BindingDB_cold/unseen_pair_setting/validate/')
    # DTI_datasets3("/stu-3031/MMDG_Data/BindingDB_cold/unseen_pair_setting/Unseen pair/train.txt",
    #               '/stu-3031/MMDG_Data/BindingDB_cold/unseen_pair_setting/train/')
    # DTI_datasets3("/stu-3031/MMDG_Data/BindingDB_cold/unseen_pair_setting/Unseen pair/test.txt", '/stu-3031/MMDG_Data/BindingDB_cold/unseen_pair_setting/test/')
    # DTI_datasets3("/stu-3031/MMDG_Data/BindingDB_cold/unseen_pair_setting/Unseen pair/dev.txt",
    #               '/stu-3031/MMDG_Data/BindingDB_cold/unseen_pair_setting/validate/')

    # #
    # DTI_datasets("GPCRs/original/GPCR_train.txt", 'GPCRs/train/')
    # DTI_datasets("GPCRs/original/GPCR_test.txt", 'GPCRs/test/')
    DTI_datasets("/stu-3031/MMDG_Data/datasets/datasets/BindingDB/original/train.txt", '/stu-3031/MMDG_Data/datasets/datasets_revise/BindingDB/train/')
    DTI_datasets("/stu-3031/MMDG_Data/datasets/datasets/BindingDB/original/test.txt", '/stu-3031/MMDG_Data/datasets/datasets_revise/BindingDB/test/')

    #
    # DTI_drugbank_datasets("/stu-3031/MMDG_Data/datasets/datasets/DrugBank/original/data.txt", '/stu-3031/MMDG_Data/datasets/datasets_revise/DrugBank/data_split/')
    # DTI_datasets("/stu-3031/MMDG_Data/datasets/datasets/Human/original/data.txt", '/stu-3031/MMDG_Data/datasets/datasets_revise/Human/data_split/')
    # DTI_datasets("/stu-3031/MMDG_Data/datasets/datasets/Celegans/original/data.txt", '/stu-3031/MMDG_Data/datasets/datasets/Celegans/data_split/')

    # DTI_drugbank_datasets("/stu-3031/MMDG_Data/datasets/datasets/Kiba/original/KIBA_DTI.txt", '/stu-3031/MMDG_Data/datasets/datasets_revise/Kiba/data_split/')
    # DTI_drugbank_datasets("/stu-3031/MMDG_Data/datasets/datasets/Davis/original/Davis_DTI.txt", '/stu-3031/MMDG_Data/datasets/datasets_revise/Davis/data_split/')

    print('The preprocess of dataset has finished!')