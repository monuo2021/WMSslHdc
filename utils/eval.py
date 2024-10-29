import os
import torch
from tqdm import tqdm 

import torch
import seaborn as sns
import torchmetrics.functional as Metrics
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def visualize_tsne(features, labels, n_classes, output_dir='./outs', image_name='tsne_visualization.png'):
	"""
    Visualizes the class hypervectors using t-SNE.

    Args:
        FM (HDFF.feature_monitor.FeatureMonitor): The FeatureMonitor object containing the class bundles.
    """
	# Ensure output directory exists
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	# 使用 t-SNE 将高维超维向量降至2维
	tsne = TSNE(n_components=2, perplexity=9, random_state=42)
	features_embedded = tsne.fit_transform(features)

	# 可视化
	plt.figure(figsize=(10, 8))
	for i in range(n_classes):
		mask = labels == i
		plt.scatter(features_embedded[mask, 0], features_embedded[mask, 1], label=f'Class {i}', alpha=0.6)

	plt.title('t-SNE Visualization of Class Hypervectors')
	plt.legend(fontsize='small')

	# 保存图片到 ./out 目录下
	output_path = os.path.join(output_dir, image_name)
	plt.savefig(output_path)
	plt.close()

def draw_confusion_matrix(all_labels, all_predictions, n_classes, matrix_title="Confusion Matrix", conf_matrix_path=None):
	conf_matrix = confusion_matrix(all_labels, all_predictions, normalize='true')
	plt.figure(figsize=(12, 10))
	sns.heatmap(
		conf_matrix, annot=True, fmt='.2%', cmap='Blues',
		xticklabels=range(n_classes), yticklabels=range(n_classes),
		annot_kws={"size": 8},  # 控制字体大小
		cbar_kws={"shrink": 0.8}  # 缩小色条
	)
	plt.xlabel('Predicted Labels', fontsize=8)
	plt.ylabel('True Labels', fontsize=8)
	plt.xticks(fontsize=6)  # 缩小 x 轴刻度标签字体
	plt.yticks(fontsize=6)  # 缩小 y 轴刻度标签字体
	plt.title(matrix_title, fontsize=10)

	# 调整色条字体大小
	colorbar = plt.gca().collections[0].colorbar
	colorbar.ax.tick_params(labelsize=6)  # 调整色条刻度字体大小

	plt.savefig(conf_matrix_path, bbox_inches='tight')
	plt.close()
	print(f'Normalized confusion matrix saved to {conf_matrix_path}')


def inference_loop(FM, dataset, device=0) -> torch.FloatTensor:
	"""Given a the feature monitor and dataset, generates OOD scores for the dataset

	Args:
		FM (HDFF.feature_monitor.FeatureMonitor): The FeatureMonitor object pre-initialised with the target model, hooks and Feature Projection Matrices
		dataset (torch.utils.data.DataLoader): Dataloader object of the corresponding dataset
		device (int, optional): Device to run inference on. Defaults to 0.

	Returns:
		FloatTensor: OOD scores per input sample
	"""
	uncertainties = torch.empty(0).to(device)

	for x, _ in tqdm(dataset):
		## Forward pass captures features for all images in the batch
		FM.model.forward(x.to(device))
		
		## Feature monitor projects the features from each layer and then bundles them across layers
		# Result is a (batch_size, HD_space_size) tensor
		feature_bundle = FM.batchFeatureBundle()

		## Compute the cosine similarity of all feature bundles to the class bundles
		# Result is a (batch_size, n_classes) tensor
		similarity = FM.VSA.similarity(feature_bundle, FM.data['class_bundles']) # (batch, n_classes)
		
		## Retrieve the raw similarities to closest class bundle (maximum cosine similarity) 
		# Result is a (batch_size) tensor
		# values, indices = torch.max(a, dim=1) -> https://pytorch.org/docs/stable/generated/torch.max.html
		closest, _ = torch.max(similarity, dim=1)

		## Invert similarities (confidence scores) to retrieve OOD scores (uncertainties)
		# 取负号 - closest，我们将最大相似度转化为不确定性分数。因为相似度越大，表示模型越自信，
		# 所以负值越小表示不确定性越低，负值越大表示不确定性越高。
		## Store these OOD scores
		uncertainties = torch.cat((uncertainties, -closest))
	
	return uncertainties

def predict_loop(FM, dataset, n_classes, image_name='tsne_BeforeRetrain.png', device=0) -> torch.FloatTensor:
	"""Given a the feature monitor and dataset, generates OOD scores for the dataset

	Args:
		FM (HDFF.feature_monitor.FeatureMonitor): The FeatureMonitor object pre-initialised with the target model, hooks and Feature Projection Matrices
		dataset (torch.utils.data.DataLoader): Dataloader object of the corresponding dataset
		device (int, optional): Device to run inference on. Defaults to 0.

	Returns:
		FloatTensor: OOD scores per input sample
	"""

	uncertainties = torch.empty(0).to(device)
	# 初始化空的张量，用于存储真实标签和预测标签
	all_features = torch.empty(0, dtype=torch.long).to(device)
	all_labels = torch.empty(0, dtype=torch.long).to(device)  # 确保标签的类型为 long
	all_predictions = torch.empty(0, dtype=torch.long).to(device)

	for x, y in tqdm(dataset):
		## Forward pass captures features for all images in the batch
		FM.model.forward(x.to(device))

		## Feature monitor projects the features from each layer and then bundles them across layers
		# Result is a (batch_size, HD_space_size) tensor
		feature_bundle = FM.batchFeatureBundle()

		## Compute the cosine similarity of all feature bundles to the class bundles
		# Result is a (batch_size, n_classes) tensor
		similarity = FM.VSA.similarity(feature_bundle, FM.data['class_bundles'])  # (batch, n_classes)

		## Retrieve the raw similarities to closest class bundle (maximum cosine similarity)
		# Result is a (batch_size) tensor
		# values, indices = torch.max(a, dim=1) -> https://pytorch.org/docs/stable/generated/torch.max.html
		closest, predictions = torch.max(similarity, dim=1)
		uncertainties = torch.cat((uncertainties, -closest))
		# 连接特征、真实标签和预测标签
		all_features = torch.cat((all_features, feature_bundle.to(device)))
		all_labels = torch.cat((all_labels, y.to(device)))  # 保存真实标签
		all_predictions = torch.cat((all_predictions, predictions.to(device)))  # 保存预测标签

	# 转换为 NumPy 数组，确保在 CPU 上进行操作
	all_features = all_features.cpu().numpy()
	all_labels = all_labels.cpu().numpy()
	all_predictions = all_predictions.cpu().numpy()

	# 计算每一类的精确率
	precision_per_class = precision_score(all_labels, all_predictions, average=None)
	# 计算整体精确率、召回率和 F1 分数
	precision = precision_score(all_labels, all_predictions, average='weighted')
	recall = recall_score(all_labels, all_predictions, average='weighted')
	f1 = f1_score(all_labels, all_predictions, average='weighted')
	# 计算整体准确率
	accuracy = accuracy_score(all_labels, all_predictions)

	print(f'每一类的精确率: {precision_per_class}')
	print(f'整体精确率: {precision:.4f}, 召回率: {recall:.4f}, F1 分数: {f1:.4f}, 准确率: {accuracy:.4f}')

	# visualize_tsne(all_features, all_predictions, n_classes, image_name=image_name)

	draw_confusion_matrix(all_labels,  # y_gt=[0,5,1,6,3,...]
						  all_predictions,  # y_pred=[0,5,1,6,3,...]
						  n_classes,
						  matrix_title="Confusion Matrix on cifar10",
						  conf_matrix_path="./outs/confusion_matrix.png")

	return uncertainties

def generate_metrics(ood_id, uncertainties, gt):
	"""
	Args:
		ood_id (int): The index of the target OOD dataset
		uncertainties (FloatTensor): Tensor of uncertainty scores for both ID and OOD set. Uncertainties are differentiated by 
		correpsonding index in gt
		gt (IntTensor): Tensor of indices corresponding to dataset IDs

	Returns:
		Dict: A dictionary containing the desired metrics
	"""
	## Binarise our gt to 0 for ID and 1 for OOD set
	temp_gt = torch.cat((gt[gt==0], gt[gt==ood_id]), dim=0)
	temp_gt = temp_gt > 0
	
	## Create new array of ID and target OOD set scores
	id_ucert = uncertainties[gt==0]
	ood_ucert = uncertainties[gt==ood_id]  
	temp_ucert = torch.cat((id_ucert, ood_ucert), dim=0)
	temp_ucert, temp_gt = temp_ucert.detach(), temp_gt.detach()

	prec, rec, thresh = Metrics.precision_recall_curve(temp_ucert, temp_gt, task="binary")
	
	## F1 Score
	f1 = 2 * (prec * rec) / (prec + rec)

	return f1, thresh

