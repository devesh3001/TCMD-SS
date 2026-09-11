"""Build the editable report and PDF exclusively from saved run artifacts."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4,landscape
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image
from src.utils.config import ROOT,load_config,output_dir


def build(config):
    out=output_dir(config);report=ROOT/"report";report.mkdir(exist_ok=True)
    read=lambda name:json.loads((out/name).read_text())
    results=read("evaluation/metrics.json");run=read("run_summary.json");training=read("diffusion/training_summary.json")
    calibration=read("calibration/calibration.json");semantic=read("semantic/metadata.json")
    audit=json.loads((ROOT/"outputs/dataset_audit.json").read_text())
    protocol=json.loads((ROOT/"outputs/splits/protocol.json").read_text())
    family=pd.read_csv(out/"evaluation/by_family.csv");classes=pd.read_csv(out/"evaluation/normal_fpr_by_class.csv")
    ablations=pd.read_csv(out/"ablation_results.csv");history=pd.read_csv(out/"diffusion/training_history.csv")
    explanations=read("heatmaps/explanations.json")
    figures=out/"figures";figures.mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,2,figsize=(11,3.5))
    axes[0].plot(history.epoch,history.loss,marker="o");axes[0].set(xlabel="Completed epoch",ylabel="Masked noise MSE",title="Figure 1a. Actual training loss")
    axes[1].plot(history.epoch,history.normal_validation_l1,marker="o");axes[1].set(xlabel="Completed epoch",ylabel="Normal validation L1",title="Figure 1b. Checkpoint selection")
    fig.tight_layout();fig.savefig(figures/"training.png",dpi=160);plt.close(fig)
    styles=getSampleStyleSheet()
    styles.add(ParagraphStyle(name="BodyCustom",fontName="Helvetica",fontSize=11,leading=15,spaceAfter=9,textColor=colors.HexColor("#23313e")))
    styles["Title"].fontSize=28;styles["Title"].leading=34
    story=[];source=[]
    def heading(text):
        story.append(Paragraph(text,styles["Heading1"]));source.extend(["", "## "+text, ""])
    def paragraph(text):
        story.append(Paragraph(text,styles["BodyCustom"]));source.extend([text,""])
    def table(frame):
        rows=[list(map(str,frame.columns))]
        for row in frame.itertuples(index=False,name=None):
            rows.append([f"{x:.4f}" if isinstance(x,float) else str(x) for x in row])
        wrapped=[[Paragraph(cell.replace("_"," "),styles["BodyText"]) for cell in row] for row in rows]
        t=Table(wrapped,repeatRows=1,hAlign="LEFT",colWidths=[730/len(frame.columns)]*len(frame.columns))
        t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#dcecf2")),("VALIGN",(0,0),(-1,-1),"TOP"),
          ("GRID",(0,0),(-1,-1),.3,colors.HexColor("#bac7d0")),("TOPPADDING",(0,0),(-1,-1),6),("BOTTOMPADDING",(0,0),(-1,-1),6)]))
        story.extend([t,Spacer(1,12)]);source.extend(["| "+" | ".join(rows[0])+" |","|"+"---|"*len(rows[0])]+["| "+" | ".join(row)+" |" for row in rows[1:]]+[""])
    def picture(path,width=730):
        from PIL import Image as PILImage
        with PILImage.open(path) as im:height=width*im.height/im.width
        story.append(Image(str(path),width=width,height=height));source.extend([f"![Figure]({path.relative_to(ROOT).as_posix()})",""])
    def page():story.append(PageBreak())
    story.append(Paragraph("TCMD-SS",styles["Title"]))
    paragraph("Terrain-Conditioned Masked Diffusion with Semantic-Spectral Scoring")
    heading("Executive summary - provisional experiment")
    paragraph("An unsupervised deep generative system learns clean Martian imagery. All eight HiRISE semantic classes are normal; terrain categories are never anomaly labels. Controlled corruptions are generated only after training and normal calibration.")
    paragraph(f"The actual run completed {training['epochs_completed']} epochs ({training['steps_completed']} optimizer steps) on {run['device']}. It used {config['size']} x {config['size']} images, a {config['base_channels']}-channel compact U-Net and {semantic['actual_model']} frozen features. The actual backbone is named here; smoke and full-scale settings are distinct.")
    table(pd.DataFrame([{k:results[k] for k in ['auroc','auprc','f1','false_positive_rate','threshold']}]))
    paragraph("These are small-sample controlled-corruption results, not a claim of deployment readiness or confirmed detection of real sensor faults. The full configuration and checkpoint-resume commands are provided.")
    page();heading("Dataset and leakage prevention")
    paragraph(f"The unchanged audited dataset contains {audit['image_count']:,} grayscale 227 x 227 JPEGs from {audit['base_landmark_count']:,} original landmarks. Archive MD5: 236d9c627db1a5970e77a01a8c8a035a. Source: https://zenodo.org/records/4002935.")
    table(pd.DataFrame(audit['class_distribution'])[['class_id','class_name','image_count','original_image_count','augmented_image_count']])
    paragraph(f"Audited base IDs, source observations and duplicate/pHash connected components are preserved. Forbidden overlap: {protocol['forbidden_overlap']}. Split file sizes: {protocol['counts']}. Calibration and test use originals; subsets are deterministically balanced across semantic classes for computational coverage, not anomaly supervision.")
    paragraph("There are no confirmed exact duplicates. Perceptual matches can be false positives, especially near black crop boundaries; the conservative grouping is retained. 'Clean' means the decoded public dataset under the stated protocol, not a human certification that every image is artifact-free.")
    page();heading("Architecture and mathematical description")
    paragraph("Figure 2. Query -> [masked conditional diffusion | frozen DINO | local frequency statistics | pooled latent statistics] -> normal-only robust calibration -> fixed fusion -> score, threshold and explanation.")
    paragraph("Diffusion: q(x_t|x_0) = sqrt(alpha_bar_t) x_0 + sqrt(1-alpha_bar_t) epsilon. The network receives x_t in the hidden mask, visible context x_0*(1-M), M and timestep t. Loss = sum(M*(epsilon_theta-epsilon)^2)/sum(M). No hidden clean query pixels enter its conditioning channel.")
    paragraph("Inference uses deterministic eta=0 DDIM updates from noise under four complementary 25% masks that cover every pixel. L1 reconstruction residuals are pooled over the highest 5% pixels. This is inpainting under visible terrain context; semantic terrain labels are not conditioning labels.")
    paragraph("Semantic: repeat grayscale to RGB, normalize with pretrained statistics, extract dense normalized DINO patch tokens, and average cosine distances to k=5 normal memory neighbors. The image score is the top-5% patch mean. Memory sampling gives each training base an equal quota and uses chunked comparisons.")
    paragraph("Spectral: local 8/16/32-pixel windows yield FFT radial-band energy, directional differences, local contrast and mean. Robust normal statistics define local deviation maps. Latent: PCA of pooled frozen tokens followed by fixed 0.1 isotropic covariance shrinkage and Mahalanobis distance. NumPy implementation avoids a Windows-blocked covariance DLL.")
    page();heading("Training and checkpoint selection")
    paragraph(f"Normal fitting-pool images: {training['training_images']}; normal validation held out: {training['validation_images']} (scored for selection: {training['validation_scored_images']}). Validation groups are removed from fitting within the training partition. Calibration and test groups are never checkpoint-selection inputs. Best normal reconstruction L1: {training['best_normal_validation_l1']:.6f}.")
    paragraph(f"Parameters: {training['parameters']:,}; diffusion steps: {config['diffusion_steps']}; DDIM sampling steps: {config['sampling_steps']}; learning rate: {config['learning_rate']}; batch size: {config['batch_size']}. AMP is enabled only with CUDA. Last/best checkpoints retain optimizer, scaler and random states for resume.")
    picture(figures/"training.png")
    paragraph(f"Measured end-to-end runtime: {run['seconds']:.1f} seconds; RAM: {run['ram_gb']:.1f} GB. Provisional bounded training is not the full convergence run.")
    page();heading("Normal-only calibration and test benchmark")
    paragraph(f"Calibration uses {calibration['normal_count']} clean originals only. Branch z = max(0, (score - median)/(1.4826*MAD + 1e-6)), capped at 100 for stability. Fixed weights are diffusion 0.40, semantic 0.25, spectral 0.20 and latent 0.15. Threshold is the normal final-score 99th percentile: {results['threshold']:.6f}. No corruption score affects fitting, stopping, weights or thresholds.")
    paragraph("Test-only generators implement horizontal/vertical stripes, dead lines/pixels, clipping, banding, local blur/noise, blank crop regions, block permutation, patch duplication, geometric tearing and synthetic foreign checkerboard patches. A local foreign-image path can be supplied explicitly. No external foreign-content dataset was used in this experiment.")
    paragraph(f"Severities actually run: {config['severities']}. Every generated record preserves source image, base/source/group IDs, type, severity, seed, generated path and change mask. Evaluation contains {results['tn']+results['fp']} clean and {results['tp']+results['fn']} corrupted examples. Metrics concern image-level detection; masks are saved for future localization evaluation.")
    table(pd.DataFrame([{k:results[k] for k in ['precision','recall','specificity','balanced_accuracy','tn','fp','fn','tp']}]))
    paragraph(f"AUROC bootstrap 95% interval: {results['auroc_ci95']}. Resampling unit is the base landmark, keeping its clean/corrupted variants together. Residual dependence between different landmarks from one observation may make this interval optimistic.")
    page();heading("Detection results by anomaly family")
    table(family[['family','auroc','auprc','f1','recall','false_positive_rate']])
    paragraph("Each family evaluation includes the same clean reference set. AUPRC and F1 depend on the artificial benchmark prevalence. Generator severity is not a real instrument failure probability.")
    table(pd.read_csv(out/"evaluation/by_severity.csv")[['severity','auroc','auprc','f1','recall']])
    page();heading("Normal false positives and ablations")
    table(classes)
    paragraph("All terrain classes remain normal. Class-specific false-positive rates are diagnostics, not an invitation to relabel rare terrain as anomalies. Small per-class denominators limit interpretation.")
    table(ablations[['variant','auroc','auprc','f1','threshold']])
    page();heading("Figure 3. Fixed-fusion ablation comparison")
    picture(out/"ablation_plot.png")
    paragraph("V1, V2 and V3 are nested scoring versions of one trained system, not three independent training runs. All variants use their own normal-only 99th-percentile threshold. Equal-weight and branch-removal ablations were specified before anomaly evaluation. No statistical superiority is asserted.")
    page();heading("Figure 4. Actual anomaly explanation")
    picture(ROOT/explanations[0]['figure'])
    paragraph(explanations[0]['explanation'])
    page();heading("Figure 5. Highest-scored clean example")
    picture(ROOT/explanations[-1]['figure'])
    paragraph(f"This is the highest-scored clean example in the evaluated subset. Flagged: {explanations[-1]['flagged']}. {explanations[-1]['explanation']} A high normal score is not evidence that its semantic terrain class is anomalous.")
    page();heading("Failure cases, limitations and final answers")
    paragraph(f"At the frozen threshold the run produced {results['fp']} false positives and {results['fn']} false negatives. Short training and limited normal sampling constrain reconstruction and statistical fidelity. Calibration size {calibration['normal_count']} is small for a 99th percentile. Synthetic patterns only approximate structural faults; there is no validated real-anomaly set.")
    paragraph("The latent score is global and contributes a uniform term to the fused heatmap; it cannot localize a defect. Heatmaps are evidence visualizations, not causal explanations or validated segmentation. Shared crop padding can influence spectral and semantic scores.")
    paragraph("Future work: run the full normal-only configuration with satellite DINO on adequate hardware, keep all thresholds fixed to normal calibration, increase independent source coverage, and evaluate separately acquired real sensor/foreign-content anomalies without using them for tuning.")
    paragraph("Final answers: the generative model runs; all four branches, calibration, test-only benchmark, ablations, explanations, notebook and report are implemented. This deliverable reports an actual provisional experiment. No full-run or deployment-level performance is claimed.")
    paragraph("Sources: NASA/JPL HiRISE dataset, Zenodo 4002935; timm/vit_small_patch16_224.dino and timm/vit_large_patch16_dinov3.sat493m model cards on Hugging Face. Local fallback DINO weights are pretrained and frozen. See README and MODEL_EVOLUTION.md for exact commands and development evidence.")
    def footer(canvas,doc):
        canvas.setFont("Helvetica",9);canvas.setFillColor(colors.HexColor("#526675"))
        canvas.drawString(40,22,"TCMD-SS | Provisional, normal-only protocol");canvas.drawRightString(800,22,str(doc.page))
    SimpleDocTemplate(str(report/"TCMD_SS_Report.pdf"),pagesize=landscape(A4),rightMargin=45,leftMargin=45,topMargin=35,bottomMargin=40).build(story,onFirstPage=footer,onLaterPages=footer)
    (report/"TCMD_SS_Report.md").write_text("\n".join(source),encoding="utf-8")
    return report/"TCMD_SS_Report.pdf"

if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--config",default="configs/local.yaml");args=parser.parse_args()
    print(build(load_config(args.config)))
