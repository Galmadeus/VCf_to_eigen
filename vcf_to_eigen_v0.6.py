import allel
import pandas as pd
import numpy as np
from time import process_time
import numba as nb


def create_ind(samples, anno, ind):
    for i in range(len(samples)):
        for k in range(len(anno)):
            if samples[i] == anno['Master_ID'][k]:
                ind['ID'][i] = samples[i]
                ind['SEX'][i] = anno['SEX'][k]
                ind['Pop'][i] = anno['Group_ID'][k]

    ind.dropna(inplace=True)
    ind.to_csv(r'samples.ind', header=None, index=None, sep='\t', mode='w')


@nb.jit(forceobj=True)
def create_chrpos(data, n):
    chr_pos = []
    chr_pos = np.array(chr_pos, dtype=np.int32)
    for i in range(len(data)):
        if data['chr'][i] == n:
            if i == 0:
                chr_pos = data['pos'][0]
            else:
                a = data['pos'][i]
                chr_pos = np.append(chr_pos, [a])

    return chr_pos


@nb.njit
def create_needed_pos(chr_pos, pos):
    needed_pos = [0] * len(chr_pos)
    for i in range(len(chr_pos)):
        for k in range(len(pos)):
            if chr_pos[i] == pos[k]:
                needed_pos[i] = pos[k]
    needed_pos = [i for i in needed_pos if i != 0]
    return needed_pos


@nb.njit
def create_mat(geno):
    # create matrix as np.uint8 (1 byte) instead of list of python integers (8 byte)
    # also no need to dynamically resize / increase list size
    geno_mat = np.zeros((len(geno[:, 0]), len(geno[1, :])), dtype=np.uint8)

    for i in np.arange(len(geno[:, 0])):
        for k in np.arange(len(geno[1, :])):
            g = geno[i, k]

            # nested ifs to avoid duplicate comparisons
            if g[0] == 0:
                if g[1] == 0:
                    geno_mat[i, k] = 2
                elif g[1] == 1:
                    geno_mat[i, k] = 1
                else:
                    geno_mat[i, k] = 9
            elif g[0] == 1:
                if g[1] == 0:
                    geno_mat[i, k] = 1
                elif g[1] == 1:
                    geno_mat[i, k] = 0
                else:
                    geno_mat[i, k] = 9
            else:
                geno_mat[i, k] = 9
    return geno_mat


def genotyping(geno, pos, chr_pos):
    needed_pos = create_needed_pos(chr_pos, pos)
    mat = create_mat(geno)
    list_difference = [item for item in chr_pos if item not in needed_pos]
    needed_pos_list = list(needed_pos)
    matrix_df = pd.DataFrame(mat, dtype=int, index=pos)
    filtered_geno_dataframe = matrix_df.loc[needed_pos_list, :]
    missing_positions_df = pd.DataFrame(index=list_difference, columns=np.arange(2054))
    missing_positions_df.fillna(2, inplace=True)
    finaldataframe = pd.concat([filtered_geno_dataframe, missing_positions_df])
    finaldataframe.sort_index(axis=0, inplace=True)
    final_mat = finaldataframe.to_numpy(dtype=int)
    return final_mat


def write_first_chr(genotype, file_name):
    with open(file_name, 'w') as fout:  # Note 'wb' instead of 'w'
        np.savetxt(fout, genotype, delimiter="", fmt='%d')
        fout.seek(-2, 2)
        fout.truncate()


def write_remaining_chr(genotype):
    with open('test_1.geno', 'a') as fout:  # Note 'wb' instead of 'w'
        np.savetxt(fout, genotype, delimiter="", fmt='%d')


if __name__ == "__main__":
    t1_start = process_time()
    data = pd.read_csv('REICH_1KG.snp', delimiter=r"\s+")
    data.columns = ['ID', "chr", "pyspos", "pos", "Ref", "Alt"]
    files = open("sample_list.txt")
    for i, line in enumerate(files):
        strip_line = line.strip()
        n = i + 1
        t2_start = process_time()
        chr_pos = create_chrpos(data, n)
        geno = allel.read_vcf(strip_line, fields=("calldata/GT",))["calldata/GT"]
        pos = allel.read_vcf(strip_line, fields=("variants/POS",))["variants/POS"]
        genotype = genotyping(geno, pos, chr_pos)
        anno = pd.read_csv('filtered_anno.txt', delimiter="\t")
        anno.columns = ['ID', 'Master_ID', 'Group_ID', 'SEX']
        ind = pd.DataFrame(index=range(2055), columns=['ID', 'SEX', 'Pop'])
        if i + 1 == 1:
            samples = allel.read_vcf(strip_line, fields=("samples",))["samples"]
            create_ind(samples, anno, ind)
            file_name = str(i + 1) + "_chr.geno"
            write_first_chr(genotype, file_name)
            t2_stop = process_time()
            print("First chromosome done")
            print("Needed time for first chr: ", t2_stop - t2_start)
            del(geno)
            del(pos)
        else:
            file_name = str(i + 1) + "_chr.geno"
            write_first_chr(genotype, file_name)
            t2_stop = process_time()
            print("Chr done:", i + 1)
            print("needed time: ", t2_stop - t2_start)
            del(geno)
            del(pos)
    print("Finished genotyping")
    t1_stop = process_time()
    print("Ennyi idő kellett teszt1:", t1_stop - t1_start)
