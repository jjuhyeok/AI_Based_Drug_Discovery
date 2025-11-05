import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt, CalcNumRings, CalcNumAmideBonds
from rdkit.Chem.AllChem import EmbedMolecule, MMFFOptimizeMolecule
from rdkit.Chem.QED import qed
from rdkit.Chem import rdPartialCharges
#from rdkit.Chem.rdmolops import Get</div>
from rdkit.Chem.Fragments import fr_benzene, fr_imidazole#, fr_triazole
from rdkit.Chem.GraphDescriptors import BertzCT # BertzCT를 위해 필요
from rdkit.Chem.rdchem import HybridizationType # HybridizationType을 위해 필요

import numpy as np
import math
from collections import defaultdict


# 3. 특정 구조적 모티프 (CYP3A4 저해 메커니즘 및 효능에 직접적 영향)
# 헴(heme)에 직접 배위 결합하는 작용기나 메커니즘 기반 저해(MBI)를 유도하는 특정 그룹의 존재는
# 강력한 CYP3A4 저해를 예측하는 데 매우 중요합니다.
# [6, 2, 12] (헴 결합 작용기)
# [13] (MBI 관련 작용기)
smarts_patterns = {
    # 헴 결합 작용기: CYP3A4 헴 철에 직접 배위 결합하여 효소 활성을 저해합니다.
    # [6, 2, 12]
    'HasPyridine': 'n1ccccc1',  # 피리딘 고리
    'HasImidazole': 'n1cn[cH]n1',  # 이미다졸 고리
    'HasTriazole': 'n1nn[cH]n1',  # 트리아졸 고리

    # 메커니즘 기반 저해 (MBI) 관련 작용기: CYP3A4에 의해 대사되어 반응성 중간체를 형성하고,
    # 이 중간체가 효소에 공유 결합하여 비가역적인 저해를 일으킵니다.
    # [13]
    'HasTertiaryAmine': '[NX3](C)(C)C',  # 3차 아민
    'HasFuran': 'o1cccc1',  # 푸란 고리
    'HasAcetylene': 'C#C',  # 아세틸렌 기능

    # 기타 중요한 작용기 (극성, 반응성, 대사 안정성 등에 영향)
    'HasAmine': '[NX3;H2,H1,H0]',  # 모든 종류의 아민 (극성 및 상호작용)
    'HasArylHalide': 'c',  # 아릴 할라이드 (반응성 및 대사 가능성)
    'HasEpoxide': 'C1OC1',  # 에폭사이드 (반응성 그룹)
    'HasAldehyde': '[CX1](=O)[H]',  # 알데하이드
    'HasKetone': '[CX2](=O)(C)C',  # 케톤
    'HasCarboxylicAcid': 'C(=O)O',  # 카르복실산
    'HasEster': 'C(=O)OC',  # 에스터
    'HasEther': 'COC',  # 에테르
    'HasSulfonamide': 'S(=O)(=O)N',  # 설폰아미드
    'HasPhosphonate': 'P(=O)(O)O',  # 포스포네이트
    'HasPhenol': 'c1ccccc1O',  # 페놀
    'HasThiol': '[SH]',  # 티올 (SMARTS 패턴 수정)
    'HasNitro': '[N+](=O)[O-]',  # 니트로 그룹
    'HasHalogen': '[F,Cl,Br,I]',  # 할로겐 원자 (SMARTS 패턴 수정)

    # 3D 형태 및 강성도에 영향을 미치는 구조
    #'HasSpiro': '*1@2*@1',  # 스피로 화합물 (3D 형태에 영향)
    'HasBridgehead': 'B',  # 브릿지헤드 원자 (강성도에 영향)
}


class CYP3A4EssentialDescriptors:
    def __init__(self):
        self.descriptor_names = []
        # SMARTS 패턴을 클래스 변수로 저장
        self.smarts_patterns = smarts_patterns

    def calculate_core_physicochemical(self, mol):
        """핵심 물리화학적 특성 - 1번 카테고리 필수"""
        descriptors = {}

        # 가장 중요한 지질친화도 지표
        descriptors['LogP'] = Descriptors.MolLogP(mol)  # CYP3A4 소수성 결합의 핵심
        descriptors['MW'] = Descriptors.MolWt(mol)      # 활성부위 크기 적합성

        # 수소 결합 - CYP3A4 특이적 결합에 필수
        descriptors['HBD'] = Descriptors.NumHDonors(mol)
        descriptors['HBA'] = Descriptors.NumHAcceptors(mol)

        # 극성 표면적 - 막투과성과 CYP3A4 결합의 균형점
        descriptors['TPSA'] = Descriptors.TPSA(mol)

        # 분자 유연성 - CYP3A4 유도적합의 핵심
        descriptors['RotatableBonds'] = Descriptors.NumRotatableBonds(mol)

        # 방향족 특성 - π-π 스택킹 상호작용
        descriptors['AromaticRings'] = Descriptors.NumAromaticRings(mol)

        return descriptors

    def calculate_critical_electronic(self, mol):
        """핵심 전자적 특성 - 2번 카테고리 중 가장 중요한 것들"""
        descriptors = {}

        # Type II 저해의 가장 중요한 지표 - 염기성 질소
        basic_nitrogens = 0
        aromatic_nitrogens = 0

        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 7:  # 질소
                if atom.GetIsAromatic():
                    aromatic_nitrogens += 1
                # 염기성 질소 추정 (헤 철과 배위 가능)
                # GetTotalNumHs() > 0: 수소가 직접 연결된 질소 (아민류)
                # (atom.GetDegree() < 3 and not atom.GetIsAromatic()): 비방향족, 3차 미만 (고리 내 질소 등)
                if atom.GetTotalNumHs() > 0 or (atom.GetDegree() < 3 and not atom.GetIsAromatic()):
                    basic_nitrogens += 1

        descriptors['BasicNitrogens'] = basic_nitrogens     # Type II 저해 핵심
        descriptors['AromaticNitrogens'] = aromatic_nitrogens # 방향족 질소의 특별한 역할

        # 전체 질소 수 - CYP3A4 저해에서 질소의 중요성
        descriptors['TotalNitrogens'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 7)

        # 산소 수 - 수소결합과 극성 상호작용
        descriptors['TotalOxygens'] = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 8)

        # 방향족 원자 비율 - π-π 상호작용의 중요성
        total_atoms = mol.GetNumHeavyAtoms()
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        descriptors['AromaticRatio'] = aromatic_atoms / total_atoms if total_atoms > 0 else 0

        return descriptors

    def calculate_key_functional_groups(self, mol):
        """핵심 기능기 - 4번 카테고리 필수 (SMARTS 패턴 사용)"""
        descriptors = {}

        # CYP3A4 저해제에서 자주 발견되는 핵심 기능기들
        # 기존 fr_ 함수 대신 smarts_patterns를 활용하여 일관성을 높임
        descriptors['Benzene_Rings'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('c1ccccc1')))
        descriptors['Imidazole_Rings'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('n1cn[cH]n1')))
        descriptors['Triazole_Rings'] = len(mol.GetSubstructMatches(Chem.MolFromSmarts('n1nn[cH]n1')))

        # 추가된 smarts_patterns의 기능기들
        for name, smarts in self.smarts_patterns.items():
            try:
                pattern = Chem.MolFromSmarts(smarts)
                if pattern is not None:
                    descriptors[name] = mol.HasSubstructMatch(pattern)
                else:
                    descriptors[name] = False
            except Exception as e:
                print(f"Error parsing SMARTS for {name} ('{smarts}'): {e}")
                descriptors[name] = False # 에러 발생 시 False로 처리

        # Lipinski 위반 - 약물성과 CYP3A4 기질성의 균형
        lipinski_violations = (
            (Descriptors.MolWt(mol) > 500) +
            (Descriptors.MolLogP(mol) > 5) +
            (Descriptors.NOCount(mol) > 10) +
            (Descriptors.NHOHCount(mol) > 5)
        )
        descriptors['Lipinski_Violations'] = lipinski_violations

        # QED - 약물유사성 (CYP3A4 기질 가능성과 연관)
        descriptors['QED'] = Descriptors.qed(mol)

        return descriptors

    def calculate_cyp3a4_specific_indicators(self, mol):
        """CYP3A4 특이적 핵심 지표들"""
        descriptors = {}

        # 1. Type II 저해 잠재력 - 헤 철 배위 가능한 질소 구조
        type2_score = 0

        # 이미다졸 패턴 (가장 강력한 Type II 저해)
        imidazole_pattern = Chem.MolFromSmarts('c1[nH]cnc1')
        if imidazole_pattern:
            type2_score += len(mol.GetSubstructMatches(imidazole_pattern)) * 3

        # 트리아졸 패턴
        triazole_pattern = Chem.MolFromSmarts('c1[nH]nnc1')
        if triazole_pattern:
            type2_score += len(mol.GetSubstructMatches(triazole_pattern)) * 3

        # 피리딘 패턴
        pyridine_pattern = Chem.MolFromSmarts('c1ccccn1')
        if pyridine_pattern:
            type2_score += len(mol.GetSubstructMatches(pyridine_pattern)) * 1

        descriptors['Type2_Inhibition_Score'] = type2_score

        # 2. CYP3A4 최적 특성 매칭 점수
        logp = Descriptors.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        tpsa = Descriptors.TPSA(mol)

        # LogP 최적 범위 (1-5, 최적 2-4)
        if 2 <= logp <= 4:
            logp_score = 1.0
        elif 1 <= logp < 2 or 4 < logp <= 5:
            logp_score = 0.5
        else:
            logp_score = 0.0

        # MW 최적 범위 (200-800, 최적 300-600)
        if 300 <= mw <= 600:
            mw_score = 1.0
        elif 200 <= mw < 300 or 600 < mw <= 800:
            mw_score = 0.5
        else:
            mw_score = 0.0

        # TPSA 최적 범위 (20-140, 최적 40-100)
        if 40 <= tpsa <= 100:
            tpsa_score = 1.0
        elif 20 <= tpsa < 40 or 100 < tpsa <= 140:
            tpsa_score = 0.5
        else:
            tpsa_score = 0.0

        descriptors['CYP3A4_Optimal_Score'] = (logp_score + mw_score + tpsa_score) / 3

        # 3. 구조적 복잡도 (CYP3A4는 복잡한 분자를 선호)
        descriptors['Complexity_Score'] = min(BertzCT(mol) / 100, 10)  # 정규화

        return descriptors

    def calculate_3d_key_properties(self, mol):
        """핵심 3D 특성만 선별"""
        descriptors = {}

        try:
            # 3D 구조 생성
            mol_copy = Chem.Mol(mol)
            EmbedMolecule(mol_copy)
            MMFFOptimizeMolecule(mol_copy)

            # 분자 부피 - CYP3A4 활성부위 적합성의 핵심
            descriptors['MolVolume'] = Descriptors.MolVol(mol_copy)

            # 관성 모멘트 비율 - 분자 형태의 핵심 지표
            pmi1 = Descriptors.PMI1(mol_copy)
            pmi2 = Descriptors.PMI2(mol_copy)
            pmi3 = Descriptors.PMI3(mol_copy)

            # 구형성 (CYP3A4 활성부위에 더 잘 맞음)
            descriptors['Spherocity'] = pmi1 / pmi3 if pmi3 > 0 else 0

            # 선형성 (너무 선형적이면 CYP3A4 결합에 불리)
            descriptors['Linearity'] = 1 - (pmi2 / pmi3) if pmi3 > 0 else 0

        except Exception:
            descriptors['MolVolume'] = 0
            descriptors['Spherocity'] = 0
            descriptors['Linearity'] = 0

        return descriptors

    def calculate_advanced_cyp3a4_features(self, mol):
        """고급 CYP3A4 특이적 특성"""
        descriptors = {}

        # 1. 헤테로고리 다양성 (CYP3A4는 다양한 헤테로고리를 인식)
        hetero_rings = 0
        for ring in Chem.GetSymmSSSR(mol):
            ring_atoms = [mol.GetAtomWithIdx(i) for i in ring]
            if any(atom.GetAtomicNum() != 6 for atom in ring_atoms):
                hetero_rings += 1

        descriptors['HeteroRings'] = hetero_rings

        # 2. 분자의 "druglikeness" vs "CYP3A4 substrate-likeness" 균형
        # 너무 drug-like하면 CYP3A4에 의해 대사될 가능성 높아짐
        qed = Descriptors.qed(mol)
        lipinski_violations = (
            (Descriptors.MolWt(mol) > 500) +
            (Descriptors.MolLogP(mol) > 5) +
            (Descriptors.NOCount(mol) > 10) +
            (Descriptors.NHOHCount(mol) > 5)
        )

        # CYP3A4 저해제는 보통 Lipinski 규칙을 일부 위반하면서도 적당한 QED를 가짐
        descriptors['Inhibitor_Balance'] = qed * (1 + lipinski_violations * 0.2)

        # 3. 방향족 밀도 (π-π 상호작용의 강도 예측)
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        total_heavy_atoms = mol.GetNumHeavyAtoms()
        descriptors['Aromatic_Density'] = aromatic_atoms / total_heavy_atoms if total_heavy_atoms > 0 else 0

        return descriptors

    def calculate_rules_beyond_lipinski(self, mol):
        """Lipinski 규칙을 넘어선 약물성 규칙들 (이미지의 Rules Beyond Lipinski)"""
        descriptors = {}

        # Lipinski Rule of 5
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)

        lipinski_violations = (
            (mw > 500) + (logp > 5) + (hbd > 5) + (hba > 10)
        )
        descriptors['Lipinski_RO5'] = lipinski_violations

        # Veber 규칙 (경구 생체이용률)
        rotatable_bonds = Descriptors.NumRotatableBonds(mol)
        tpsa = Descriptors.TPSA(mol)
        veber_violations = (rotatable_bonds > 10) + (tpsa > 140)
        descriptors['Veber_Rule'] = veber_violations

        # Egan 규칙 (흡수 예측)
        descriptors['Egan_LogP'] = logp
        descriptors['Egan_PSA'] = tpsa
        egan_violations = (logp > 5.88) + (tpsa > 131.6)
        descriptors['Egan_Rule'] = egan_violations

        # Ghose 규칙
        atom_count = mol.GetNumAtoms()
        molar_refractivity = Descriptors.MolMR(mol)
        ghose_violations = (
            (mw < 160 or mw > 480) +
            (logp < -0.4 or logp > 5.6) +
            (atom_count < 20 or atom_count > 70) +
            (molar_refractivity < 40 or molar_refractivity > 130)
        )
        descriptors['Ghose_Rule'] = ghose_violations

        # Muegge 규칙
        rings = Descriptors.RingCount(mol)
        carbons = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
        heteroatoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [1, 6])

        muegge_violations = (
            (mw < 200 or mw > 600) +
            (logp < -2 or logp > 5) +
            (tpsa > 150) +
            (rings > 7) +
            (carbons < 5) +
            (heteroatoms < 1 or heteroatoms > 15) +
            (rotatable_bonds > 15) +
            (hbd > 5) +
            (hba > 10)
        )
        descriptors['Muegge_Rule'] = muegge_violations

        return descriptors

    def calculate_qed_components(self, mol):
        """QED (Quantitative Estimate of Drug-likeness) 구성 요소들"""
        descriptors = {}

        # QED 전체 점수
        descriptors['QED_Total'] = Descriptors.qed(mol)

        # QED 구성 요소별 계산
        mw = Descriptors.MolWt(mol)
        logp = Descriptors.MolLogP(mol)
        hbd = Descriptors.NumHDonors(mol)
        hba = Descriptors.NumHAcceptors(mol)
        tpsa = Descriptors.TPSA(mol)
        rotb = Descriptors.NumRotatableBonds(mol)
        arom = Descriptors.NumAromaticRings(mol)
        alerts = Descriptors.NumAliphaticRings(mol)  # 구조적 경고 대신 지방족 고리 사용

        # 각 파라미터의 바람직도 함수 (단순화)
        descriptors['QED_MW'] = max(0, 1 - abs(mw - 350) / 350)
        descriptors['QED_LogP'] = max(0, 1 - abs(logp - 2.5) / 5)
        descriptors['QED_HBD'] = max(0, 1 - hbd / 5)
        descriptors['QED_HBA'] = max(0, 1 - hba / 10)
        descriptors['QED_PSA'] = max(0, 1 - tpsa / 140)
        descriptors['QED_RotB'] = max(0, 1 - rotb / 10)
        descriptors['QED_Arom'] = min(1, arom / 3)
        descriptors['QED_Alerts'] = max(0, 1 - alerts / 5)

        return descriptors

    def calculate_labute_asa(self, mol):
        """Labute ASA (Accessible Surface Area) 관련 특성"""
        descriptors = {}

        # 전체 Labute ASA
        descriptors['LabuteASA_Total'] = Descriptors.LabuteASA(mol)

        # 원자별 ASA 기여도 계산
        try:
            # 각 원자의 ASA 기여도 (근사값)
            asa_contributions = []
            for atom in mol.GetAtoms():
                # Van der Waals 반지름 기반 ASA 추정 (RDKit의 GetVdwRadius는 현재 지원되지 않음)
                # 직접 VdW 반지름을 매핑하거나, 더 정교한 계산 방식 사용 필요. 여기서는 예시를 위해 단순화
                vdw_radii = {1: 1.20, 6: 1.70, 7: 1.55, 8: 1.52, 9: 1.47, 15: 1.80, 16: 1.80}
                radius = vdw_radii.get(atom.GetAtomicNum(), 2.0) # 기본값 2.0
                surface_area = 4 * math.pi * radius ** 2 # 구의 표면적 공식 (간단화)
                asa_contributions.append(surface_area)

            descriptors['LabuteASA_Mean'] = np.mean(asa_contributions)
            descriptors['LabuteASA_Std'] = np.std(asa_contributions)

            # 극성/비극성 ASA 분리
            polar_asa = sum(asa for i, asa in enumerate(asa_contributions)
                            if mol.GetAtomWithIdx(i).GetAtomicNum() in [7, 8, 9, 15, 16])
            nonpolar_asa = sum(asa_contributions) - polar_asa

            descriptors['LabuteASA_Polar'] = polar_asa
            descriptors['LabuteASA_Nonpolar'] = nonpolar_asa
            descriptors['LabuteASA_PolarRatio'] = polar_asa / sum(asa_contributions) if sum(asa_contributions) > 0 else 0

        except Exception:
            descriptors['LabuteASA_Mean'] = 0
            descriptors['LabuteASA_Std'] = 0
            descriptors['LabuteASA_Polar'] = 0
            descriptors['LabuteASA_Nonpolar'] = 0
            descriptors['LabuteASA_PolarRatio'] = 0

        return descriptors

    def calculate_asphericity(self, mol):
        """분자의 비구형성 (CalcAsphericity) 계산"""
        descriptors = {}

        try:
            # 3D 구조 생성
            mol_copy = Chem.Mol(mol)
            EmbedMolecule(mol_copy)
            MMFFOptimizeMolecule(mol_copy)

            # 관성 모멘트 계산
            pmi1 = Descriptors.PMI1(mol_copy)
            pmi2 = Descriptors.PMI2(mol_copy)
            pmi3 = Descriptors.PMI3(mol_copy)

            # Asphericity 계산 (0: 완전 구형, 1: 완전 선형)
            if (pmi1 + pmi2 + pmi3) > 0: # 분모가 0이 아닌 경우에만 계산
                asphericity = ((pmi3 - pmi2)**2 + (pmi2 - pmi1)**2 + (pmi1 - pmi3)**2) / (2 * (pmi1 + pmi2 + pmi3)**2)
                descriptors['Asphericity'] = asphericity

                # 추가 형태 지수들
                descriptors['Acylindricity'] = pmi2 - pmi1
                descriptors['InertialShapeIndex'] = pmi2 / pmi3 if pmi3 > 0 else 0
                # eccentricity 계산 시 pmi3가 0이 아닐 때만 math.sqrt 호출
                descriptors['Eccentricity'] = math.sqrt(1 - (pmi1**2 / pmi3**2)) if pmi3 > 0 and pmi3**2 != 0 else 0
            else:
                descriptors['Asphericity'] = 0
                descriptors['Acylindricity'] = 0
                descriptors['InertialShapeIndex'] = 0
                descriptors['Eccentricity'] = 0

        except Exception:
            descriptors['Asphericity'] = 0
            descriptors['Acylindricity'] = 0
            descriptors['InertialShapeIndex'] = 0
            descriptors['Eccentricity'] = 0

        return descriptors

    def calculate_partial_charges_advanced(self, mol):
        """EPartialCharge 고급 부분 전하 특성"""
        descriptors = {}

        try:
            # Gasteiger 전하 계산
            rdPartialCharges.ComputeGasteigerCharges(mol)

            charges = []
            positive_charges = []
            negative_charges = []

            for atom in mol.GetAtoms():
                try:
                    charge = float(atom.GetProp('_GasteigerCharge'))
                    if not math.isnan(charge):
                        charges.append(charge)
                        if charge > 0:
                            positive_charges.append(charge)
                        elif charge < 0:
                            negative_charges.append(charge)
                except Exception:
                    continue

            if charges:
                descriptors['PartialCharge_Max'] = max(charges)
                descriptors['PartialCharge_Min'] = min(charges)
                descriptors['PartialCharge_Range'] = max(charges) - min(charges)
                descriptors['PartialCharge_Mean'] = np.mean(charges)
                descriptors['PartialCharge_Std'] = np.std(charges)

                # 양전하/음전하 분석
                descriptors['PartialCharge_PositiveSum'] = sum(positive_charges)
                descriptors['PartialCharge_NegativeSum'] = sum(negative_charges)
                descriptors['PartialCharge_PositiveCount'] = len(positive_charges)
                descriptors['PartialCharge_NegativeCount'] = len(negative_charges)

                # 전하 분포의 비대칭성
                # 분모가 0이 되는 경우 방지
                std_dev = np.std(charges)
                descriptors['PartialCharge_Skewness'] = np.mean([(c - np.mean(charges))**3 for c in charges]) / (std_dev**3) if std_dev > 0 else 0
                descriptors['PartialCharge_Kurtosis'] = np.mean([(c - np.mean(charges))**4 for c in charges]) / (std_dev**4) if std_dev > 0 else 0

            else:
                for key in ['PartialCharge_Max', 'PartialCharge_Min', 'PartialCharge_Range',
                            'PartialCharge_Mean', 'PartialCharge_Std', 'PartialCharge_PositiveSum',
                            'PartialCharge_NegativeSum', 'PartialCharge_PositiveCount',
                            'PartialCharge_NegativeCount', 'PartialCharge_Skewness', 'PartialCharge_Kurtosis']:
                    descriptors[key] = 0

        except Exception:
            for key in ['PartialCharge_Max', 'PartialCharge_Min', 'PartialCharge_Range',
                        'PartialCharge_Mean', 'PartialCharge_Std', 'PartialCharge_PositiveSum',
                        'PartialCharge_NegativeSum', 'PartialCharge_PositiveCount',
                        'PartialCharge_NegativeCount', 'PartialCharge_Skewness', 'PartialCharge_Kurtosis']:
                descriptors[key] = 0

        return descriptors

    def calculate_atomic_environment_counts(self, mol):
        """원자 환경 카운트 (Atomic Environment Counts) - 이미지의 두 번째 특성"""
        descriptors = {}

        # 기본 원자 환경 카운트
        atom_env_counts = defaultdict(int)

        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            degree = atom.GetDegree()
            hybridization = str(atom.GetHybridization())
            is_aromatic = atom.GetIsAromatic()
            formal_charge = atom.GetFormalCharge()

            # 다양한 환경 특성 조합
            env_key = f"{atomic_num}_{degree}_{hybridization}_{is_aromatic}_{formal_charge}"
            atom_env_counts[env_key] += 1

            # 간단한 환경 카운트들
            descriptors[f'AtomCount_{atomic_num}'] = descriptors.get(f'AtomCount_{atomic_num}', 0) + 1
            descriptors[f'Degree_{degree}_Count'] = descriptors.get(f'Degree_{degree}_Count', 0) + 1

            if is_aromatic:
                descriptors[f'Aromatic_{atomic_num}_Count'] = descriptors.get(f'Aromatic_{atomic_num}_Count', 0) + 1

        # 결합 환경 카운트
        bond_env_counts = defaultdict(int)

        for bond in mol.GetBonds():
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            bond_type = str(bond.GetBondType())
            is_aromatic = bond.GetIsAromatic()
            is_ring = bond.IsInRing()

            # 결합 환경 특성
            bond_key = f"{begin_atom.GetAtomicNum()}_{end_atom.GetAtomicNum()}_{bond_type}_{is_aromatic}_{is_ring}"
            bond_env_counts[bond_key] += 1

            # 간단한 결합 카운트들
            descriptors[f'Bond_{bond_type}_Count'] = descriptors.get(f'Bond_{bond_type}_Count', 0) + 1

            if is_aromatic:
                descriptors['AromaticBond_Count'] = descriptors.get('AromaticBond_Count', 0) + 1
            if is_ring:
                descriptors['RingBond_Count'] = descriptors.get('RingBond_Count', 0) + 1

        # 고리 환경 분석
        ring_info = mol.GetRingInfo()
        for ring in ring_info.AtomRings():
            ring_size = len(ring)
            descriptors[f'Ring_Size_{ring_size}_Count'] = descriptors.get(f'Ring_Size_{ring_size}_Count', 0) + 1

            # 고리 내 원자 타입 분석
            ring_atoms = [mol.GetAtomWithIdx(i) for i in ring]
            aromatic_in_ring = sum(1 for atom in ring_atoms if atom.GetIsAromatic())
            hetero_in_ring = sum(1 for atom in ring_atoms if atom.GetAtomicNum() != 6)

            if aromatic_in_ring == ring_size:
                descriptors[f'AromaticRing_Size_{ring_size}_Count'] = descriptors.get(f'AromaticRing_Size_{ring_size}_Count', 0) + 1
            if hetero_in_ring > 0:
                descriptors[f'HeteroRing_Size_{ring_size}_Count'] = descriptors.get(f'HeteroRing_Size_{ring_size}_Count', 0) + 1

        # 환경 다양성 지수
        descriptors['AtomEnv_Diversity'] = len(atom_env_counts)
        descriptors['BondEnv_Diversity'] = len(bond_env_counts)

        # 가장 빈번한 환경들
        if atom_env_counts:
            most_common_atom_env = max(atom_env_counts.values())
            descriptors['MostCommon_AtomEnv_Count'] = most_common_atom_env
            descriptors['AtomEnv_Dominance'] = most_common_atom_env / sum(atom_env_counts.values())
        else:
            descriptors['MostCommon_AtomEnv_Count'] = 0
            descriptors['AtomEnv_Dominance'] = 0

        # 특정 중요 환경들 (CYP3A4 관련)
        # sp2 탄소 (방향족 시스템)
        sp2_carbons = sum(1 for atom in mol.GetAtoms()
                          if atom.GetAtomicNum() == 6 and
                          atom.GetHybridization() == HybridizationType.SP2)
        descriptors['SP2_Carbon_Count'] = sp2_carbons

        # sp3 탄소 (지방족 부분)
        sp3_carbons = sum(1 for atom in mol.GetAtoms()
                          if atom.GetAtomicNum() == 6 and
                          atom.GetHybridization() == HybridizationType.SP3)
        descriptors['SP3_Carbon_Count'] = sp3_carbons

        # 질소 환경 상세 분석 (CYP3A4 저해에 중요)
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 7:  # 질소
                degree = atom.GetDegree()
                is_aromatic = atom.GetIsAromatic()
                hybridization = atom.GetHybridization()

                if is_aromatic and hybridization == HybridizationType.SP2:
                    descriptors['AromaticSP2_Nitrogen_Count'] = descriptors.get('AromaticSP2_Nitrogen_Count', 0) + 1
                elif not is_aromatic and hybridization == HybridizationType.SP3:
                    descriptors['AliphaticSP3_Nitrogen_Count'] = descriptors.get('AliphaticSP3_Nitrogen_Count', 0) + 1

        return descriptors

    def generate_comprehensive_descriptors(self, smiles_list):
        """
        모든 서술자를 포함한 포괄적 버전 (이미지의 모든 특성 포함)

        Parameters:
        -----------
        smiles_list : list
            SMILES 문자열 리스트

        Returns:
        --------
        pandas.DataFrame
            모든 서술자를 포함하는 포괄적 데이터프레임
        """
        all_descriptors = []
        failed_molecules = []

        for i, smiles in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    failed_molecules.append((i, smiles, "Invalid SMILES"))
                    continue

                # 수소 추가
                mol = Chem.AddHs(mol)

                # 모든 서술자 계산
                mol_descriptors = {}

                # 기존 핵심 서술자들
                mol_descriptors.update(self.calculate_core_physicochemical(mol))
                mol_descriptors.update(self.calculate_critical_electronic(mol))
                mol_descriptors.update(self.calculate_key_functional_groups(mol))
                mol_descriptors.update(self.calculate_cyp3a4_specific_indicators(mol))
                mol_descriptors.update(self.calculate_3d_key_properties(mol))
                mol_descriptors.update(self.calculate_advanced_cyp3a4_features(mol))

                # 이미지에서 추가된 새로운 서술자들
                mol_descriptors.update(self.calculate_rules_beyond_lipinski(mol))
                mol_descriptors.update(self.calculate_qed_components(mol))
                mol_descriptors.update(self.calculate_labute_asa(mol))
                mol_descriptors.update(self.calculate_asphericity(mol))
                mol_descriptors.update(self.calculate_partial_charges_advanced(mol))
                mol_descriptors.update(self.calculate_atomic_environment_counts(mol))

                # SMILES 추가
                mol_descriptors['SMILES'] = smiles
                all_descriptors.append(mol_descriptors)

            except Exception as e:
                failed_molecules.append((i, smiles, str(e)))
                continue

        if failed_molecules:
            print(f"Failed to process {len(failed_molecules)} molecules:")
            for i, smiles, error in failed_molecules[:5]:
                print(f"  Index {i}: {smiles} - {error}")

        # 데이터프레임 생성
        df = pd.DataFrame(all_descriptors)

        # 무한대나 NaN 값을 0으로 대체
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        print(f"Successfully generated {len(df.columns)-1} COMPREHENSIVE descriptors for {len(df)} molecules")
        self._print_comprehensive_descriptor_summary()

        return df

    def _print_comprehensive_descriptor_summary(self):
        """포괄적 서술자 요약 정보 출력"""
        print("\n=== CYP3A4 Comprehensive Descriptors Summary ===")
        print("1. Core Physicochemical (7): LogP, MW, HBD, HBA, TPSA, RotatableBonds, AromaticRings")
        print("2. Critical Electronic (5): BasicNitrogens, AromaticNitrogens, TotalNitrogens, TotalOxygens, AromaticRatio")
        print("3. Key Functional Groups (5 + SMARTS Patterns): Benzene/Imidazole/Triazole_Rings, Lipinski_Violations, QED, plus all SMARTS patterns (e.g., HasPyridine, HasTertiaryAmine, HasFuran, HasThiol, HasHalogen, etc.)")
        print("4. CYP3A4 Specific (3): Type2_Inhibition_Score, CYP3A4_Optimal_Score, Complexity_Score")
        print("5. 3D Properties (3): MolVolume, Spherocity, Linearity")
        print("6. Advanced CYP3A4 (3): HeteroRings, Inhibitor_Balance, Aromatic_Density")
        print("7. Rules Beyond Lipinski (9): Lipinski, Veber, Egan, Ghose, Muegge rules")
        print("8. QED Components (8): QED_Total and component scores")
        print("9. Labute ASA (6): Total, Mean, Std, Polar, Nonpolar, PolarRatio")
        print("10. Asphericity (4): Asphericity, Acylindricity, InertialShapeIndex, Eccentricity")
        print("11. Partial Charges (11): Max, Min, Range, Mean, Std, Positive/Negative sums/counts, Skewness, Kurtosis")
        print("12. Atomic Environment Counts (Variable): Detailed atomic and bond environment analysis")
        print("\nTotal: 60+ comprehensive descriptors covering all aspects from the images")
        print("===============================================")

    def generate_essential_descriptors(self, smiles_list):
        """
        핵심 CYP3A4 관련 서술자만 생성 (총 ~30개 내외)

        Parameters:
        -----------
        smiles_list : list
            SMILES 문자열 리스트

        Returns:
        --------
        pandas.DataFrame
            핵심 서술자만 포함하는 경량화된 데이터프레임
        """
        all_descriptors = []
        failed_molecules = []

        for i, smiles in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol is None:
                    failed_molecules.append((i, smiles, "Invalid SMILES"))
                    continue

                # 수소 추가
                mol = Chem.AddHs(mol)

                # 핵심 서술자들만 계산
                mol_descriptors = {}
                mol_descriptors.update(self.calculate_core_physicochemical(mol))      # 7개
                mol_descriptors.update(self.calculate_critical_electronic(mol))      # 5개
                mol_descriptors.update(self.calculate_key_functional_groups(mol))     # 5 + SMARTS patterns
                mol_descriptors.update(self.calculate_cyp3a4_specific_indicators(mol)) # 3개
                mol_descriptors.update(self.calculate_3d_key_properties(mol))         # 3개
                mol_descriptors.update(self.calculate_advanced_cyp3a4_features(mol))  # 3개

                # SMILES 추가
                mol_descriptors['SMILES'] = smiles
                all_descriptors.append(mol_descriptors)

            except Exception as e:
                failed_molecules.append((i, smiles, str(e)))
                continue

        if failed_molecules:
            print(f"Failed to process {len(failed_molecules)} molecules:")
            for i, smiles, error in failed_molecules[:5]:
                print(f"  Index {i}: {smiles} - {error}")

        # 데이터프레임 생성
        df = pd.DataFrame(all_descriptors)

        # 무한대나 NaN 값을 0으로 대체
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

        print(f"Successfully generated {len(df.columns)-1} ESSENTIAL descriptors for {len(df)} molecules")
        self._print_descriptor_summary()

        return df

    def _print_descriptor_summary(self):
        """생성된 서술자 요약 정보 출력"""
        print("\n=== CYP3A4 Essential Descriptors Summary ===")
        print("1. Core Physicochemical (7):")
        print("   - LogP, MW, HBD, HBA, TPSA, RotatableBonds, AromaticRings")
        print("\n2. Critical Electronic (5):")
        print("   - BasicNitrogens, AromaticNitrogens, TotalNitrogens, TotalOxygens, AromaticRatio")
        print("\n3. Key Functional Groups (5 + SMARTS Patterns):")
        print("   - Benzene_Rings, Imidazole_Rings, Triazole_Rings, Lipinski_Violations, QED, plus selected SMARTS patterns")
        print("\n4. CYP3A4 Specific Indicators (3):")
        print("   - Type2_Inhibition_Score, CYP3A4_Optimal_Score, Complexity_Score")
        print("\n5. Key 3D Properties (3):")
        print("   - MolVolume, Spherocity, Linearity")
        print("\n6. Advanced CYP3A4 Features (3):")
        print("   - HeteroRings, Inhibitor_Balance, Aromatic_Density")
        print("\nTotal: Approximately 26-30 essential descriptors (variable due to SMARTS patterns).")
        print("===============================================")


def example_usage():
    """핵심 서술자 생성기 사용 예제 - 포괄적 버전 포함"""

    # CYP3A4 저해제/비저해제 예제
    cyp3a4_inhibitors = [
        'CN1CCN(CC1)C2=C(C=C3C(=C2)N=CN=C3NC4=CC(=C(C=C4)F)Cl)OC',  # 게피티니브 (강한 저해)
        'CC1=C(C=CC(=C1)C)C(=O)NCC2=NN=C(S2)C3=CC=CC=C3',          # 케토코나졸 유사체
        'CCN(CC)CCNC(=O)C1=CC(=C(C=C1)N)S(=O)(=O)N',           # 설피리드
    ]

    cyp3a4_non_inhibitors = [
        'CC(C)CC1=CC=C(C=C1)C(C)C(=O)O',      # 이부프로펜 (약한 저해)
        'CC(C)(C)NCC(C1=CC(=CC=C1)O)O',        # 살부타몰
        'C1=CC=C(C=C1)CC(C(=O)O)N',            # 페닐알라닌
    ]

    all_smiles = cyp3a4_inhibitors + cyp3a4_non_inhibitors

    # 핵심 서술자 생성기 사용
    generator = CYP3A4EssentialDescriptors()

    print("=== Testing Essential Descriptors (26 features) ===")
    df_essential = generator.generate_essential_descriptors(all_smiles)
    print(f"Essential DataFrame shape: {df_essential.shape}")

    print("\n=== Testing Comprehensive Descriptors (60+ features) ===")
    df_comprehensive = generator.generate_comprehensive_descriptors(all_smiles)
    print(f"Comprehensive DataFrame shape: {df_comprehensive.shape}")

    # 새로운 특성들 중 일부 확인
    print("\n=== New Features from Images ===")
    new_features = ['Lipinski_RO5', 'Veber_Rule', 'QED_Total', 'QED_MW', 'LabuteASA_Total',
                    'Asphericity', 'PartialCharge_Range', 'AtomEnv_Diversity', 'SP2_Carbon_Count']

    # SMARTS 패턴 기반 특성 추가
    smarts_based_features = ['HasPyridine', 'HasImidazole', 'HasTriazole', 'HasTertiaryAmine', 'HasFuran', 'HasThiol']
    new_features.extend(smarts_based_features)


    for feature in new_features:
        if feature in df_comprehensive.columns:
            # 불리언 값은 직접 출력하고, 숫자 값은 포맷팅
            if isinstance(df_comprehensive[feature].iloc[0], bool):
                print(f"{feature}: {df_comprehensive[feature].iloc[0]}")
            else:
                print(f"{feature}: {df_comprehensive[feature].iloc[0]:.3f}")

    # 저해제 vs 비저해제 비교 (포괄적 버전)
    inhibitor_data = df_comprehensive.iloc[:len(cyp3a4_inhibitors)]  # 저해제
    non_inhibitor_data = df_comprehensive.iloc[len(cyp3a4_inhibitors):]  # 비저해제

    print("\n=== Key Differences (Comprehensive Analysis) ===")
    key_features = ['Type2_Inhibition_Score', 'QED_Total', 'Lipinski_RO5', 'Asphericity',
                    'PartialCharge_Range', 'AtomEnv_Diversity', 'AromaticSP2_Nitrogen_Count']
    key_features.extend(['HasPyridine', 'HasImidazole', 'HasTriazole', 'HasTertiaryAmine']) # 추가 SMARTS 패턴

    for feature in key_features:
        if feature in df_comprehensive.columns:
            if isinstance(df_comprehensive[feature].iloc[0], bool):
                # 불리언 값은 True/False count 또는 비율로 비교
                inh_true_count = inhibitor_data[feature].sum()
                non_inh_true_count = non_inhibitor_data[feature].sum()
                print(f"{feature}: Inhibitors_True_Count={inh_true_count}, Non-inhibitors_True_Count={non_inh_true_count}")
            else:
                inh_mean = inhibitor_data[feature].mean()
                non_inh_mean = non_inhibitor_data[feature].mean()
                print(f"{feature}: Inhibitors={inh_mean:.3f}, Non-inhibitors={non_inh_mean:.3f}")


if __name__ == "__main__":
    example_usage()