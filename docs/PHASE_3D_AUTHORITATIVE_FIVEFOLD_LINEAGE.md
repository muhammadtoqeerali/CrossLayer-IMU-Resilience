# Phase 3D Authoritative Five-Fold Lineage Audit

## Status

PASS

## KFold calls

- Source: `/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master/dataloaders/KFoldDataloader.py`
- Line: 25
- Call: `KFold(n_splits=self.k, shuffle=True, random_state=42)`

- Source: `/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto_master/dataloaders/KFoldDataloader.py`
- Line: 25
- Call: `KFold(n_splits=self.k, shuffle=True, random_state=42)`

- Source: `/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori_copy/dataloaders/KFoldDataloader.py`
- Line: 16
- Call: `KFold(n_splits=self.k, shuffle=True, random_state=42)`

- Source: `/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori_copy/Protechto-master_ori/dataloaders/KFoldDataloader.py`
- Line: 16
- Call: `KFold(n_splits=self.k, shuffle=True, random_state=42)`

## train_test_split calls

- Source: `/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master/dataloaders/KFoldDataloader.py`
- Line: 38
- Call: `train_test_split(train_idx, test_size=0.2, random_state=42)`

- Source: `/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto_master/dataloaders/KFoldDataloader.py`
- Line: 38
- Call: `train_test_split(train_idx, test_size=0.2, random_state=42)`

- Source: `/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori_copy/dataloaders/KFoldDataloader.py`
- Line: 25
- Call: `train_test_split(
            train_subjects_index, test_size=0.2, random_state=42
        )`

- Source: `/mnt/hdd16T/ToqeerHomeBackup/toqeer/Protechto-master_ori_copy/Protechto-master_ori/dataloaders/KFoldDataloader.py`
- Line: 25
- Call: `train_test_split(
            train_subjects_index, test_size=0.2, random_state=42
        )`

## 400-ms dataset roots

- `/mnt/hdd16T/protechto/data/KFall_oriented/segments/400ms_50ov_npseg_filt_binary`: 32 subjects, 999=False, 1000=False
- `/mnt/hdd16T/protechto/data/OnField/segments/400ms_50ov_npseg_filt_binary`: 12 subjects, 999=True, 1000=True
- `/mnt/hdd16T/protechto/data/UniVrFall_KFall_OF/segments/400ms_50ov_npseg_filt_binary`: 73 subjects, 999=True, 1000=True
- `/mnt/hdd16T/protechto/data/UniVrFall_oriented/segments/400ms_50ov_npseg_filt_binary`: 29 subjects, 999=False, 1000=False
- `/mnt/hdd16T/protechto/data/back/UniVrFall/segments/400ms_50ov_npseg_filt_binary`: 37 subjects, 999=True, 1000=True
- `/mnt/hdd16T/protechto/data/back/UniVrFall_KFall/segments/400ms_50ov_npseg_filt_binary`: 69 subjects, 999=True, 1000=True

## Fold checkpoint evidence

Candidate fold runs: 0

Complete FP32 five-fold runs: 0

Complete quantized five-fold runs: 0

## Open timing issue

Official UniVR fall events: 573

Local historical onset+impact annotation rows: 574

Status: OPEN

No event-time metric is frozen until the one-row discrepancy is reconciled.

## Methodological separation

The recovered 47/6/14 split remains checkpoint provenance.

The five-fold protocol is a separate fold-specific retraining protocol.
