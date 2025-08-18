# -*- coding: utf-8 -*-
import os
import sys
import pandas as pd
import numpy as np
from modlamp.sequences import Random
import difflib
import argparse
from modlamp.descriptors import PeptideDescriptor, GlobalDescriptor

from collections import Counter
import re
import sys
from glob import glob


def generate_fasta_from_csv(sequence_csv_path, fasta_path):
    df = pd.read_csv(sequence_csv_path, header=0)
    ddd = df.iloc[:, 0].apply(str.upper).tolist()
    with open(fasta_path, 'w') as f:
        for index, value in enumerate(ddd, start=1):
            f.write(">gi_" + str(index) + "\n" + value + "\n")


def generate_peptide_features(sequence_csv_path, save_path, input_path):
    # 生成FASTA文件路径
    base_name = os.path.basename(sequence_csv_path).split(".")[0]
    fasta_path = os.path.join(input_path, f"{base_name}.fasta")
    generate_fasta_from_csv(sequence_csv_path, fasta_path)

    # 计算肽水平特征
    pepdesc_gi = PeptideDescriptor(fasta_path, 'eisenberg')
    pepdesc_gi.calculate_global()  # 全局Eisenberg疏水性
    pepdesc_gi.calculate_moment(append=True)
    pepdesc_gi.load_scale('gravy')
    pepdesc_gi.calculate_global(append=True)  # 全局GRAVY疏水性
    pepdesc_gi.calculate_moment(append=True)  # GRAVY疏水矩
    pepdesc_gi.load_scale('z3')
    pepdesc_gi.calculate_autocorr(1, append=True)
    pepdesc_gi.load_scale('z5')
    pepdesc_gi.calculate_autocorr(1, append=True)
    pepdesc_gi.load_scale('AASI')
    pepdesc_gi.calculate_global(append=True)
    pepdesc_gi.calculate_moment(append=True)
    pepdesc_gi.load_scale('ABHPRK')
    pepdesc_gi.calculate_global(append=True)
    pepdesc_gi.load_scale('argos')
    pepdesc_gi.calculate_global(append=True)
    pepdesc_gi.calculate_moment(append=True)
    pepdesc_gi.load_scale('bulkiness')
    pepdesc_gi.calculate_global(append=True)
    pepdesc_gi.calculate_moment(append=True)
    pepdesc_gi.load_scale('charge_phys')
    pepdesc_gi.calculate_global(append=True)
    pepdesc_gi.load_scale('charge_acid')
    pepdesc_gi.calculate_global(append=True)
    pepdesc_gi.load_scale('Ez')
    pepdesc_gi.calculate_global(append=True)
    pepdesc_gi.load_scale('flexibility')
    pepdesc_gi.calculate_global(append=True)
    pepdesc_gi.calculate_moment(append=True)
    pepdesc_gi.load_scale('grantham')
    pepdesc_gi.calculate_global(append=True)
    pepdesc_gi.load_scale('hopp-woods')
    pepdesc_gi.calculate_global(append=True)
    pepdesc_gi.calculate_moment(append=True)
    pepdesc_gi.load_scale('ISAECI')
    pepdesc_gi.calculate_global(append=True)
    pepdesc_gi.load_scale('janin')
    pepdesc_gi.calculate_global(append=True)
    pepdesc_gi.calculate_moment(append=True)
    pepdesc_gi.load_scale('kytedoolittle')
    pepdesc_gi.calculate_global(append=True)
    pepdesc_gi.calculate_moment(append=True)
    pepdesc_gi.load_scale('levitt_alpha')
    pepdesc_gi.calculate_global(append=True)
    pepdesc_gi.calculate_moment(append=True)
    pepdesc_gi.load_scale('MSS')
    pepdesc_gi.calculate_global(append=True)
    pepdesc_gi.calculate_moment(append=True)
    pepdesc_gi.load_scale('MSW')
    pepdesc_gi.calculate_global(append=True)
    pepdesc_gi.load_scale('pepArc')
    pepdesc_gi.calculate_global(append=True)
    pepdesc_gi.load_scale('pepcats')
    pepdesc_gi.calculate_global(append=True)
    pepdesc_gi.load_scale('polarity')
    pepdesc_gi.calculate_global(append=True)
    pepdesc_gi.calculate_moment(append=True)
    pepdesc_gi.load_scale('PPCALI')
    pepdesc_gi.calculate_global(append=True)
    pepdesc_gi.load_scale('refractivity')
    pepdesc_gi.calculate_global(append=True)
    pepdesc_gi.calculate_moment(append=True)
    pepdesc_gi.load_scale('t_scale')
    pepdesc_gi.calculate_global(append=True)
    pepdesc_gi.load_scale('TM_tend')
    pepdesc_gi.calculate_global(append=True)
    pepdesc_gi.calculate_moment(append=True)

    # 保存肽水平特征
    col_names1 = 'ID,Sequence,H_Eisenberg,uH_Eisenberg,H_GRAVY,uH_GRAVY,Z3_1,Z3_2,Z3_3,Z5_1,Z5_2,Z5_3,Z5_4,Z5_5,S_AASI,uS_AASI,modlas_ABHPRK,H_argos,uH_argos,B_Builkiness,uB_Builkiness,charge_phys,charge_acid,Ez,flexibility,u_flexibility,Grantham,H_HoppWoods,uH-HoppWoods,ISAECI,H_Janin,uH_Janin,H_KyteDoolittle,uH_KyteDoolittle,F_Levitt,uF_Levitt,MSS_shape,u_MSS_shape,MSW,pepArc,pepcats,polarity,u_polarity,PPCALI,refractivity,u_refractivity,t_scale,TM_tend,u_TM_tend'
    pep_fea_path = os.path.join(save_path, 'moldamp_pepfea.csv')
    pepdesc_gi.save_descriptor(pep_fea_path, header=col_names1)
    print(f"肽水平特征已保存至: {pep_fea_path}")

    # 计算全局特征
    globdesc_gi = GlobalDescriptor(fasta_path)
    globdesc_gi.length()  # 序列长度
    globdesc_gi.boman_index(append=True)  # Boman指数
    globdesc_gi.aromaticity(append=True)  # 芳香性
    globdesc_gi.aliphatic_index(append=True)  # 脂肪族指数
    globdesc_gi.instability_index(append=True)  # 不稳定性指数
    globdesc_gi.calculate_charge(ph=7.4, amide=False, append=True)  # 净电荷
    globdesc_gi.calculate_MW(amide=False, append=True)  # 分子量
    globdesc_gi.isoelectric_point(amide=False, append=True)  # 等电点
    globdesc_gi.hydrophobic_ratio(append=True)  # 疏水性比例

    # 保存全局特征
    col_names2 = 'ID,Sequence,Length,BomanIndex,Aromaticity,AliphaticIndex,InstabilityIndex,NetCharge,MW,IsoelectricPoint,HydrophobicRatio'
    glob_fea_path = os.path.join(save_path, 'moldamp_gloabfea.csv')
    globdesc_gi.save_descriptor(glob_fea_path, header=col_names2)
    print(f"全局特征已保存至: {glob_fea_path}")

    # 合并两个CSV文件
    pep_fea_df = pd.read_csv(pep_fea_path, header=0, index_col=None)
    glob_fea_df = pd.read_csv(glob_fea_path, header=0, index_col=None)

    pepfea = pep_fea_df.iloc[:, 2:]
    globalfea = glob_fea_df.iloc[:, 2:]
    sequence = pep_fea_df.iloc[:, 1]

    ori = pd.read_csv(sequence_csv_path, header=0)
    label = ori.iloc[:, 1]
    sequence_ori = ori.iloc[:, 0]

    result = pd.concat([sequence_ori, label, sequence, pepfea, globalfea], axis=1, ignore_index=True)
    name = os.path.basename(sequence_csv_path).split(".")[0]

    result.to_csv(os.path.join(save_path, f"{name}_modlampfea.csv"), index_label=False, index=False)
    print("merged!!")


class CTDCalculator:
    def __init__(self):
        # 氨基酸分组字典 - 基于多种物理化学性质
        self.group1 = {
            'hydrophobicity_PRAM900101': 'RKEDQN',
            'hydrophobicity_ARGP820101': 'QSTNGDE',
            'hydrophobicity_ZIMJ680101': 'QNGSWTDERA',
            'hydrophobicity_PONP930101': 'KPDESNQT',
            'hydrophobicity_CASG920101': 'KDEQPSRNTG',
            'hydrophobicity_ENGD860101': 'RDKENQHYP',
            'hydrophobicity_FASG890101': 'KERSQD',
            'normwaalsvolume': 'GASTPDC',
            'polarity': 'LIFWCMVY',
            'polarizability': 'GASDT',
            'charge': 'KR',
            'secondarystruct': 'EALMQKRH',
            'solventaccess': 'ALFCGIVW'
        }

        self.group2 = {
            'hydrophobicity_PRAM900101': 'GASTPHY',
            'hydrophobicity_ARGP820101': 'RAHCKMV',
            'hydrophobicity_ZIMJ680101': 'HMCKV',
            'hydrophobicity_PONP930101': 'GRHA',
            'hydrophobicity_CASG920101': 'AHYMLV',
            'hydrophobicity_ENGD860101': 'SGTAW',
            'hydrophobicity_FASG890101': 'NTPG',
            'normwaalsvolume': 'NVEQIL',
            'polarity': 'PATGS',
            'polarizability': 'CPNVEQIL',
            'charge': 'ANCQGHILMFPSTWYV',
            'secondarystruct': 'VIYCWFT',
            'solventaccess': 'RKQEND'
        }

        self.group3 = {
            'hydrophobicity_PRAM900101': 'CLVIMFW',
            'hydrophobicity_ARGP820101': 'LYPFIW',
            'hydrophobicity_ZIMJ680101': 'LPFYI',
            'hydrophobicity_PONP930101': 'YMFWLCVI',
            'hydrophobicity_CASG920101': 'FIWC',
            'hydrophobicity_ENGD860101': 'CVLIMF',
            'hydrophobicity_FASG890101': 'AYHWVMFLIC',
            'normwaalsvolume': 'MHKFRYW',
            'polarity': 'HQRKNED',
            'polarizability': 'KMHFRYW',
            'charge': 'DE',
            'secondarystruct': 'GNPSD',
            'solventaccess': 'MSPTHY'
        }

        self.properties = (
            'hydrophobicity_PRAM900101', 'hydrophobicity_ARGP820101', 'hydrophobicity_ZIMJ680101',
            'hydrophobicity_PONP930101', 'hydrophobicity_CASG920101', 'hydrophobicity_ENGD860101',
            'hydrophobicity_FASG890101', 'normwaalsvolume', 'polarity', 'polarizability',
            'charge', 'secondarystruct', 'solventaccess'
        )

        # 分布特征的百分比位点
        self.distribution_points = [0, 25, 50, 75, 100]

    def _count_aa_in_group(self, sequence, aa_group):
        return sum(1 for aa in sequence if aa in aa_group)

    def calculate_composition(self, sequence):
        if not sequence:
            return pd.Series()

        length = len(sequence)
        ctdc = {}

        for prop in self.properties:
            c1 = self._count_aa_in_group(sequence, self.group1[prop]) / length
            c2 = self._count_aa_in_group(sequence, self.group2[prop]) / length
            c3 = self._count_aa_in_group(sequence, self.group3[prop]) / length

            total = c1 + c2 + c3
            if total > 0:
                c1, c2, c3 = c1 / total, c2 / total, c3 / total

            ctdc[f'{prop}.G1_CTD.C'] = c1
            ctdc[f'{prop}.G2_CTD.C'] = c2
            ctdc[f'{prop}.G3_CTD.C'] = c3

        return pd.Series(ctdc)

    def calculate_transition(self, sequence):
        if len(sequence) < 2:
            return pd.Series()

        aa_pairs = [sequence[i:i + 2] for i in range(len(sequence) - 1)]
        pair_count = len(aa_pairs)
        ctdt = {}

        for prop in self.properties:
            tr12 = 0  # Group1 <-> Group2
            tr13 = 0  # Group1 <-> Group3
            tr23 = 0  # Group2 <-> Group3

            for pair in aa_pairs:
                aa1, aa2 = pair
                g1 = aa1 in self.group1[prop]
                g2 = aa1 in self.group2[prop]
                g3 = aa1 in self.group3[prop]
                g1_ = aa2 in self.group1[prop]
                g2_ = aa2 in self.group2[prop]
                g3_ = aa2 in self.group3[prop]

                if (g1 and g2_) or (g2 and g1_):
                    tr12 += 1
                elif (g1 and g3_) or (g3 and g1_):
                    tr13 += 1
                elif (g2 and g3_) or (g3 and g2_):
                    tr23 += 1

            ctdt[f'{prop}.Tr1221'] = tr12 / pair_count if pair_count > 0 else 0
            ctdt[f'{prop}.Tr1331'] = tr13 / pair_count if pair_count > 0 else 0
            ctdt[f'{prop}.Tr2332'] = tr23 / pair_count if pair_count > 0 else 0

        return pd.Series(ctdt)

    def calculate_distribution(self, sequence):
        if not sequence:
            return pd.Series()

        length = len(sequence)
        ctdd = {}

        for prop in self.properties:
            for group_idx, group in enumerate([self.group1, self.group2, self.group3], 1):
                group_aa = group[prop]
                positions = [i + 1 for i, aa in enumerate(sequence) if aa in group_aa]

                if not positions:
                    for point in self.distribution_points:
                        ctdd[f'{prop}.G{group_idx}.residue{point}'] = 0
                    continue

                first = (positions[0] / length) * 100
                last = (positions[-1] / length) * 100

                p25_pos = int(len(positions) * 0.25)
                p50_pos = int(len(positions) * 0.50)
                p75_pos = int(len(positions) * 0.75)

                p25 = (positions[p25_pos] / length) * 100 if len(positions) > 1 else first
                p50 = (positions[p50_pos] / length) * 100 if len(positions) > 2 else first
                p75 = (positions[p75_pos] / length) * 100 if len(positions) > 3 else last

                ctdd[f'{prop}.G{group_idx}.residue{0}'] = first
                ctdd[f'{prop}.G{group_idx}.residue{25}'] = p25
                ctdd[f'{prop}.G{group_idx}.residue{50}'] = p50
                ctdd[f'{prop}.G{group_idx}.residue{75}'] = p75
                ctdd[f'{prop}.G{group_idx}.residue{100}'] = last

        return pd.Series(ctdd)

    def calculate_ctd(self, sequence):
        if not sequence:
            return pd.Series()

        clean_sequence = re.sub('[^ACDEFGHIKLMNPQRSTVWY]', '', sequence)

        ctdc = self.calculate_composition(clean_sequence)
        ctdt = self.calculate_transition(clean_sequence)
        ctdd = self.calculate_distribution(clean_sequence)

        ctd = pd.concat([ctdc, ctdt, ctdd])
        return ctd

    def calculate_ctd_for_sequences(self, sequences, labels=None):
        if not sequences:
            return pd.DataFrame()

        if labels is None:
            labels = [None] * len(sequences)

        ctd_features_list = []

        for label, seq in zip(labels, sequences):
            ctd_features = self.calculate_ctd(seq)
            if not ctd_features.empty:
                ctd_features = pd.concat([
                    pd.Series({'label': label, 'sequence': seq}),
                    ctd_features
                ])
                ctd_features_list.append(ctd_features)

        if not ctd_features_list:
            return pd.DataFrame()

        ctd_df = pd.DataFrame(ctd_features_list)

        cols = ['label', 'sequence'] + [col for col in ctd_df.columns if col not in ['label', 'sequence']]
        ctd_df = ctd_df[cols]

        return ctd_df


def process_csv(input_file, output_file):
    try:
        df = pd.read_csv(input_file,header=0)

        if 'sequence' not in df.columns or 'label' not in df.columns:
            raise ValueError("CSV文件必须包含'sequence'和'label'列")

        sequences = df['sequence'].tolist()
        labels = df['label'].tolist()

        ctd_calculator = CTDCalculator()

        print(f"正在计算 {len(sequences)} 条序列的CTD特征...")
        ctd_df = ctd_calculator.calculate_ctd_for_sequences(sequences, labels)

        ctd_df.to_csv(output_file, index=False)
        print(f"CTD特征已保存到 {output_file}")
        print(f"输出数据形状: {ctd_df.shape}")

    except Exception as e:
        print(f"处理CSV文件时出错: {str(e)}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Generate sequence representations.")
    parser.add_argument('input', type=str, help="Input CSV file containing sequences.")
    parser.add_argument('output_root', type=str, help="Output CSV file to save representations.")

    args = parser.parse_args()
    csv_file=args.input
    print(f"\n===== 开始处理文件: {csv_file} =====")
    file_name = os.path.splitext(os.path.basename(csv_file))[0]
    output_dir = args.output_root
    os.makedirs(output_dir, exist_ok=True)
    # 生成肽特征
    generate_peptide_features(csv_file, output_dir, output_dir)
    print(f"{file_name} 的moldamp特征生成完成")
    # 生成CTD特征
    ctd_output = os.path.join(output_dir, f"{file_name}_CTD.csv")
    process_csv(csv_file, ctd_output)
    # 合并所有特征
    modfea_path = os.path.join(output_dir, f"{file_name}_modlampfea.csv")
    ctdfea_path = ctd_output
    # 读取特征文件
    ctdfea = pd.read_csv(ctdfea_path, header=0)
    modfea = pd.read_csv(modfea_path, header=0)
    # 提取特征部分
    modfeaa = modfea.iloc[:, 3:]
    # s_l_s = modfea.iloc[:, :3]
    ctdfeaa = ctdfea.iloc[:, 2:]
    s_l = ctdfea.iloc[:, :2]
    # 合并并保存
    resultt = pd.concat([modfeaa, ctdfeaa,s_l], axis=1)
    allfea_path = os.path.join(output_dir, "fea1.csv")
    resultt.to_csv(allfea_path, index=False)
    print(f"{file_name} 的所有特征已合并至: {allfea_path}")
    print(f"===== 文件 {csv_file} 处理完成 =====")

    print(f"\n所有CSV文件处理完成，存储于 {args.output_root}")


if __name__ == "__main__":
    main()
