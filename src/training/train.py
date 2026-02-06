from typing import Optional, Dict, Any
import torch
import pandas as pd
import os
import yaml
from tqdm.auto import tqdm
import wandb
import json

from ..utils.logging import logger
from ..utils.ner_models import SimpleNERModel
from ..utils.ner_datasets import TransformersNERDataset
from ..utils.ner_preprocessing import (
    load_texts_to_dict,
    predict_and_extract_entities,
    load_ann_to_dataframe,
    build_tagged_df,
    create_label_dicts,
)
from ..utils.metrics import (
    compute_span_prf_global,
    compute_span_prf_by_label,
)

def train_model(
    data_path: str,
) -> None:
    
    config_path = os.path.join(
        data_path,
        "config",
        "parameters",
        "ner.yml",
    )

    if not os.path.exists(config_path):
        logger.error(f"NER config file not found: {config_path}")
        raise FileNotFoundError(f"Missing config file: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg: Dict[str, Any] = yaml.safe_load(f)

    if cfg is None:
        logger.error("NER config file is empty or malformed.")
        raise ValueError("Invalid ner.yml configuration")
    
    if "ner" not in cfg:
        logger.error("Missing 'ner' section in ner.yml")
        raise KeyError("ner.yml must contain a top-level 'ner' section")
    
    os.environ["WANDB_MODE"] = "online"
    os.environ["WANDB_DIR"] = "/tmp/wandb"         
    os.environ["WANDB_CACHE_DIR"] = "/tmp/wandb"
    
    wandb_tags = [
    "ner",
    "span-level",
    "iob"
    ]

    out_dir = "/tmp/wandb_exports"
    os.makedirs(out_dir, exist_ok=True)

    raw_train_path = os.path.join(
        data_path,
        cfg["ner"]["data"]["raw_train"],
    )    
    raw_test_path = os.path.join(
        data_path,
        cfg["ner"]["data"]["raw_test"],
    )
    
    raw_ann_df = load_ann_to_dataframe(raw_train_path)
    raw_test_ann_df = load_ann_to_dataframe(raw_test_path)

    id2label, label2id = create_label_dicts(raw_ann_df.label.unique().tolist())

    train_texts_dict = load_texts_to_dict(raw_train_path)
    test_texts_dict = load_texts_to_dict(raw_test_path)

    logger.info(
        "NER model config => "
        f"base_model={cfg['ner']['ner_model']['base_model']} | "
    )


    ner_model = SimpleNERModel(
        model_name=cfg["ner"]["ner_model"]["base_model"],
        num_tag=len(id2label)
    )

    tokenizer = ner_model.get_tokenizer()
    sub_tok_max = ner_model.get_max_len(cfg["ner"]["ner_model"].get("max_length", 128))

    raw_train_df = build_tagged_df(
        data_path=raw_train_path,
        tokenizer=tokenizer,
        sub_tok_max=sub_tok_max,
        ann_df=raw_ann_df,
        label2id=label2id,
    )

    raw_test_df = build_tagged_df(
        data_path=raw_test_path,
        tokenizer=tokenizer,
        sub_tok_max=sub_tok_max,
        ann_df=raw_test_ann_df,
        label2id=label2id,
    )

    
    logger.info(f"RAW train documents: {len(train_texts_dict)}")
    logger.info(f"RAW train fragments: {len(raw_train_df)}")

    if cfg["ner"]["data"].get("use_augmented", False):
        valid_labels = set(raw_ann_df["label"].unique().tolist())

        augmented_train_path = os.path.join(
            data_path,
            cfg["ner"]["data"]["augmented_root"],
            cfg["ner"]["prompt"],
            cfg["ner"]["llm_model"],  
        )
        logger.info(f"Augmented train path: {augmented_train_path}")
        wandb_tags.extend([
            "augmented",
            f"prompt:{cfg['ner']['prompt']}",
            f"llm:{cfg['ner']['llm_model']}",
        ])
        aug_suffix = cfg["ner"]["augmented"].get("suffix", "_aug")
        aug_texts = load_texts_to_dict(augmented_train_path)
        logger.info(f"AUGMENTED train documents: {len(aug_texts)}")
        aug_ann_df = load_ann_to_dataframe(augmented_train_path)
        aug_ann_df = aug_ann_df[aug_ann_df["label"].isin(valid_labels)].reset_index(drop=True)
        aug_train_df = build_tagged_df(
            data_path=augmented_train_path,
            tokenizer=tokenizer,
            sub_tok_max=sub_tok_max,
            ann_df=aug_ann_df,
            label2id=label2id,
        )
        logger.info(f"AUGMENTED train fragments: {len(aug_train_df)}")

        aug_train_df["filename"] = aug_train_df["filename"].astype(str) + aug_suffix
        aug_texts = {f"{k}{aug_suffix}": v for k, v in aug_texts.items()}
        train_texts_dict.update(aug_texts)
        before = len(raw_train_df)
        raw_train_df = pd.concat([raw_train_df, aug_train_df], ignore_index=True)
        raw_ann_df = pd.concat([raw_ann_df, aug_ann_df], ignore_index=True)
        after = len(raw_train_df)

        logger.info(
            f"RAW train fragments after adding augmented: {before} -> {after} (+{after - before})"
        )
        logger.info(
            f"Unique documents in merged train_df: {raw_train_df['filename'].nunique()}"
        )
        logger.info(raw_train_df .head())
    else:
        augmented_train_path = None
        wandb_tags.append("no-augmented")
        logger.info("Augmented data disabled")

    wandb.init(
        project="ToxHabitsAugmented_Subtask1",
        entity="fgallegodonoso-barcelona-supercomputing-center-centro-na",
        mode="online",
        tags=wandb_tags,
        settings=wandb.Settings(
            save_code=False,
            disable_git=True,
        ),
        config={
            "use_augmented": cfg["ner"]["data"].get("use_augmented", False),
            "prompt": cfg["ner"].get("prompt"),
            "llm_model": cfg["ner"].get("llm_model"),
            "augmented_suffix": cfg["ner"]["augmented"].get("suffix", "_aug"),
            "base_model": cfg["ner"]["ner_model"]["base_model"],
            "max_length": cfg["ner"]["ner_model"]["max_length"],
            "batch_size": cfg["ner"]["training"]["batch_size"],
            "learning_rate": cfg["ner"]["training"]["learning_rate"],
            "epochs": cfg["ner"]["training"].get("epochs", 25),
            "label_pad_id": cfg["ner"]["training"]["label_pad_id"],
        },
    )

    train_ds = TransformersNERDataset(
        raw_train_df,
        tokenizer,
        int(cfg["ner"]["ner_model"].get("max_length", 128)),
        label_pad_id=int(cfg["ner"]["training"].get("label_pad_id", -100)),
    )
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=int(cfg["ner"]["training"].get("batch_size", 8)),
        shuffle=True      
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ner_model.to(device)

    optimizer = torch.optim.AdamW(
        ner_model.parameters(),
        lr=float(cfg["ner"]["training"].get("learning_rate", 5e-5)),
    )

    loss_fn = torch.nn.CrossEntropyLoss(
        ignore_index=int(cfg["ner"]["training"].get("label_pad_id", -100)),
    )

    logger.info(
        "Training config => "
        f"epochs={cfg['ner']['training'].get('epochs', 25)} | "
        f"batch_size={cfg['ner']['training'].get('batch_size', 8)} | "
        f"lr={cfg['ner']['training'].get('learning_rate', 5e-5)} | "
        f"label_pad_id={cfg['ner']['training'].get('label_pad_id', -100)}"
    )



    for epoch in range(1, int(cfg['ner']['training'].get('epochs', 25)+1)):
        ner_model.train()
        pbar = tqdm(
            train_loader,
            desc=f"[Epoch {epoch}/{cfg['ner']['training'].get('epochs', 25)}] Training",
            leave=False,
        )
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            optimizer.zero_grad()
            outputs = ner_model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs.view(-1, outputs.shape[-1]), labels.view(-1))
            loss.backward()
            optimizer.step()


        train_entities_df = predict_and_extract_entities(
            df=raw_train_df,
            model=ner_model,
            id2label=id2label,
            texts_dict=train_texts_dict,   
            device=device,
            max_len=cfg["ner"]["ner_model"]["max_length"],
            batch_size=cfg["ner"]["training"]["batch_size"],
        )
        global_metrics = compute_span_prf_global(
            pred_df=train_entities_df,
            gold_df=raw_ann_df,
        )
        logger.info(
            f"[TRAIN] Global | "
            f"P={global_metrics['precision']:.4f} "
            f"R={global_metrics['recall']:.4f} "
            f"F1={global_metrics['f1']:.4f} "
            f"(TP={global_metrics['tp']} FP={global_metrics['fp']} FN={global_metrics['fn']})"
        )
        wandb.log(
        {
            "P-Overall (Train)": global_metrics["precision"],
            "R-Overall (Train)": global_metrics["recall"],
            "F1-Overall (Train)": global_metrics["f1"],
            "TP-Overall (Train)": global_metrics["tp"],
            "FP-Overall (Train)": global_metrics["fp"],
            "FN-Overall (Train)": global_metrics["fn"],
            "epoch": epoch,
        },
        step=epoch,
    )
        by_label_df = compute_span_prf_by_label(
            pred_df=train_entities_df,
            gold_df=raw_ann_df,
        )

        for _, r in by_label_df.iterrows():
            lab = r["label"]
            wandb.log(
                {
                    f"F1-{lab} (Train)": r["f1"],
                    f"P-{lab} (Train)": r["precision"],
                    f"R-{lab} (Train)": r["recall"],
                    f"TP-{lab} (Train)": global_metrics["tp"],
                    f"FP-{lab} (Train)": global_metrics["fp"],
                    f"FN-{lab} (Train)": global_metrics["fn"],
                },
                step=epoch,
            )
        logger.info("[TRAIN] Per-label metrics:")
        logger.info("\n" + by_label_df.to_string(index=False))


    pred_path = os.path.join(out_dir, "train_predictions.tsv")
    gold_path = os.path.join(out_dir, "train_gold.tsv")

    test_pred_path = os.path.join(out_dir, "predictions.tsv")
    test_gold_path = os.path.join(out_dir, "gold.tsv")
    label2id_path = os.path.join(out_dir, "label2id.json")

    artifact = wandb.Artifact(
        name="outputs",
        type="dataset",
        metadata={
            "what": "Final train predictions (entity-level), gold standard, and label2id mapping",
            "note": "Files are TSV (pred/gold) and JSON (label2id).",
        },
    )
    train_entities_df.to_csv(pred_path, sep="\t", index=False)
    raw_ann_df.to_csv(gold_path, sep="\t", index=False)

    with open(label2id_path, "w", encoding="utf-8") as f:
        json.dump(label2id, f, ensure_ascii=False, indent=2)


    test_entities_df = predict_and_extract_entities(
        df=raw_test_df,
        model=ner_model,
        id2label=id2label,
        texts_dict=test_texts_dict,   
        device=device,
        max_len=cfg["ner"]["ner_model"]["max_length"],
        batch_size=cfg["ner"]["training"]["batch_size"],
    )
    global_metrics = compute_span_prf_global(
        pred_df=test_entities_df,
        gold_df=raw_test_ann_df,
    )
    logger.info(
        f"[TEST] Global | "
        f"P={global_metrics['precision']:.4f} "
        f"R={global_metrics['recall']:.4f} "
        f"F1={global_metrics['f1']:.4f} "
        f"(TP={global_metrics['tp']} FP={global_metrics['fp']} FN={global_metrics['fn']})"
    )
    wandb.log(
    {
        "P-Overall (Test)": global_metrics["precision"],
        "R-Overall (Test)": global_metrics["recall"],
        "F1-Overall (Test)": global_metrics["f1"],
        "TP-Overall (Test)": global_metrics["tp"],
        "FP-Overall (Test)": global_metrics["fp"],
        "FN-Overall (Test)": global_metrics["fn"],
        "epoch": epoch,
    },
    step=epoch,
)
    by_label_df = compute_span_prf_by_label(
        pred_df=test_entities_df,
        gold_df=raw_test_ann_df,
    )

    for _, r in by_label_df.iterrows():
        lab = r["label"]
        wandb.log(
            {
                f"F1-{lab} (Test)": r["f1"],
                f"P-{lab} (Test)": r["precision"],
                f"R-{lab} (Test)": r["recall"],
                f"TP-{lab} (Test)": global_metrics["tp"],
                f"FP-{lab} (Test)": global_metrics["fp"],
                f"FN-{lab} (Test)": global_metrics["fn"],
            },
        )
    logger.info("[TEST] Per-label metrics:")
    logger.info("\n" + by_label_df.to_string(index=False))
    test_entities_df.to_csv(test_pred_path, sep="\t", index=False)
    raw_test_ann_df.to_csv(test_gold_path, sep="\t", index=False)

    artifact.add_file(test_gold_path)
    artifact.add_file(test_pred_path)

    artifact.add_file(pred_path)
    artifact.add_file(gold_path)

    artifact.add_file(label2id_path)
    wandb.log_artifact(artifact)
    wandb.finish()