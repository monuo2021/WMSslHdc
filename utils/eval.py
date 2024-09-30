import torch
from tqdm import tqdm 

import torch
import torchmetrics.functional as Metrics
from sklearn.metrics import precision_score, recall_score, f1_score

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


def predict_loop(FM, dataset, device=0) -> torch.FloatTensor:
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
		# 连接真实标签和预测标签
		all_labels = torch.cat((all_labels, y.to(device)))  # 保存真实标签
		all_predictions = torch.cat((all_predictions, predictions.to(device)))  # 保存预测标签

	# 转换为 NumPy 数组，确保在 CPU 上进行操作
	all_labels = all_labels.cpu().numpy()
	all_predictions = all_predictions.cpu().numpy()

	# 计算精确率、召回率和 F1 分数
	precision = precision_score(all_labels, all_predictions, average='weighted')
	recall = recall_score(all_labels, all_predictions, average='weighted')
	f1 = f1_score(all_labels, all_predictions, average='weighted')

	print(f'精确率: {precision:.4f}, 召回率: {recall:.4f}, F1 分数: {f1:.4f}')

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

