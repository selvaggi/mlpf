
import torch

def load_train_model(args, dev):
    from src.models.GATr.Gatr_pf_e_noise import ExampleWrapper as GravnetModel
    model = GravnetModel.load_from_checkpoint(
        args.load_model_weights, args=args, dev=0, map_location=dev,strict=False)
    return model 


def load_test_model(args, dev):
    if args.load_model_weights is not None and args.correction:
            from src.models.GATr.Gatr_pf_e_noise import ExampleWrapper as GravnetModel
            model = GravnetModel.load_from_checkpoint(
                args.load_model_weights, args=args, dev=0, map_location=dev, strict=False
            )
            model2 = GravnetModel.load_from_checkpoint(args.load_model_weights_clustering, args=args, dev=0, strict=False, map_location=torch.device("cuda:3")) # Load the good clustering
            model.gatr = model2.gatr
            model.ScaledGooeyBatchNorm2_1 = model2.ScaledGooeyBatchNorm2_1
            model.clustering = model2.clustering
            model.beta = model2.beta
    return model
