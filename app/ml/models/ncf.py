import torch
import torch.nn as nn

class NCF(nn.Module):
    
    """
    Neural Collaborative Filtering
    
    GMF (Generalized Matrix Factorization) + MLP 결합
    
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64,
        mlp_layers: list = [128,64,32],
        dropout: float = 0.2
    ):
        
        super(NCF,self).__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        #GML Part - Element-wise product
        self.user_embedding_gmf = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_gmf = nn.Embedding(n_items, embedding_dim)
        
        # MLP Part - concatenation
        self.user_embedding_mlp = nn.Embedding(n_users, embedding_dim)
        self.item_embedding_mlp = nn.Embedding(n_items, embedding_dim)
        
        mlp_modules = []
        input_size = embedding_dim * 2
        
        for layer_size in mlp_layers:
            mlp_modules.append(nn.Linear(input_size,layer_size))
            mlp_modules.append(nn.ReLU())
            mlp_modules.append(nn.Dropout(dropout))
            input_size = layer_size
            
        self.mlp = nn.Sequential(*mlp_modules)
        
        #Final prediction layer
        self.predict_layer = nn.Linear(
            embedding_dim + mlp_layers[-1],
            1
        )
        
        self._init_weight()
        
    def _init_weight(self ):
        
        # Xavier 초기화
        
        nn.init.xavier_uniform_(self.user_embedding_gmf.weight)
        nn.init.xavier_uniform_(self.item_embedding_gmf.weight)
        nn.init.xavier_uniform_(self.user_embedding_mlp.weight)
        nn.init.xavier_uniform_(self.item_embedding_mlp.weight)
        
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                
        nn.init.xavier_uniform_(self.predict_layer.weight)
        
    def forward(self, user_ids, item_ids):
        
        """
        Args:
            user_ids: (batch_size,)
            item_ids: (batch_size,)
            
        Returns:
            predictions: (batch_size,) - 1 ~ 10 점수
            
        """
        
        # GMF Part
        
        user_emb_gmf = self.user_embedding_gmf(user_ids)
        item_emb_gmf = self.item_embedding_gmf(item_ids)
        gmf_output = user_emb_gmf * item_emb_gmf
        
        # MLP Part
        user_emb_mlp = self.user_embedding_mlp(user_ids)
        item_emb_mlp = self.item_embedding_mlp(item_ids)
        mlp_input = torch.cat([user_emb_mlp, item_emb_mlp], dim=1)
        mlp_output = self.mlp(mlp_input)
        
        # Concatenate GMF and MLP
        ncf_concat = torch.cat([gmf_output, mlp_output],dim=1)
        
        # Final prediction (1~10)
        prediction = self.predict_layer(ncf_concat)
        
        # Sigmoid으로 0 ~ 1, 그다음 1 ~ 10으로 스케일링
        prediction = torch.sigmoid(prediction) * 9 + 1
        
        return prediction.squeeze()
    
    
class SimpleMF(nn.Module):
    
    """
    Simple Matrix Factorization (비교용 베이스라인)
    """
    
    def __init__(
        self,
        n_users: int,
        n_items: int,
        embedding_dim: int = 64
        ):
    
        super(SimpleMF, self).__init__()
    
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        self.user_bias = nn.Embedding(n_users, 1)
        self.item_bias = nn.Embedding(n_items, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.item_bias.weight)
        
    def forward(self, user_ids, item_ids):
        user_emb = self.user_embedding(user_ids)
        item_emb = self.item_embedding(item_ids)
        
        # Dot product
        dot_product = (user_emb * item_emb).sum(dim=1)
        
        # Add biases
        prediction = (
            dot_product
            + self.user_bias(user_ids).squeeze()
            + self.item_bias(item_ids).squeeze()
            + self.global_bias
        )
        
        # Clip to 1 ~ 10
        prediction = torch.clamp(prediction, 1.0, 10.0)
        
        return prediction
    
    