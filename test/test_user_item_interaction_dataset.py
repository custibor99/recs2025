import pytest
import torch
import pandas as pd

from matrix_factorization.datasets import UserItemInteractionDataset

# Mock data for testing
@pytest.fixture
def setup_data():
    data = {
        'client_id': [1, 2, 3, 1, 2],
        'sku': [101, 102, 103, 104, 105]
    }
    user_map = {1: 0, 2: 1, 3: 2}
    item_map = {101: 0, 102: 1, 103: 2, 104: 3, 105: 4}
    category_map = {101: 'A', 102: 'B', 103: 'C', 104: 'D', 105: 'E'}
    
    dataset = UserItemInteractionDataset(
        pd.DataFrame(data), user_map, item_map, category_map, num_negatives=2
    )
    return dataset


def test_len(setup_data):
    dataset = setup_data
    assert len(dataset) == 5  # Should be 5 since there are 5 user-item pairs


def test_getitem_valid_index(setup_data):
    dataset = setup_data
    sample = dataset[0]
    assert len(sample) == 3  # Expect 'user', 'pos_item', and 'neg_items'
    assert isinstance(sample["user"], torch.Tensor)
    assert isinstance(sample["pos_item"], torch.Tensor)
    assert isinstance(sample["neg_items"], torch.Tensor)


def test_getitem_invalid_index(setup_data):
    dataset = setup_data
    with pytest.raises(IndexError):
        dataset[10]  # Should raise an IndexError


def test_negative_sampling(setup_data):
    dataset = setup_data
    sample = dataset[0]
    neg_items = sample["neg_items"].tolist()
    assert len(neg_items) == 2  # Should generate exactly 2 negative items
    assert sample["pos_item"].item() not in neg_items  # The positive item should not be in negative samples


def test_negative_sampling_no_valid_negatives():
    small_df = pd.DataFrame({
        'client_id': [1],
        'sku': [101]
    })
    
    small_user_map = {1: 0}
    small_item_map = {101: 0}
    small_category_map = {101: 'A'}
    
    small_dataset = UserItemInteractionDataset(small_df, small_user_map, small_item_map, small_category_map, num_negatives=1)
    
    sample = small_dataset[0]  # This should return a sample, even if it has no negative items
    assert sample["neg_items"].numel() == 0  # No negative samples found, should return an empty tensor


def test_getitem_edge_case():
    df_single_item = pd.DataFrame({
        'client_id': [1],
        'sku': [101]
    })
    
    single_item_user_map = {1: 0}
    single_item_map = {101: 0}
    single_item_category_map = {101: 'A'}
    
    single_item_dataset = UserItemInteractionDataset(df_single_item, single_item_user_map, single_item_map, single_item_category_map, num_negatives=1)
    
    sample = single_item_dataset[0]
    
    assert sample["neg_items"].numel() == 0  # Should return an empty tensor since there are no negative items


def test_large_data():
    large_df = pd.DataFrame({
        'client_id': [i for i in range(1000)],
        'sku': [i for i in range(1000, 2000)]
    })
    
    large_user_map = {i: i for i in range(1000)}
    large_item_map = {i: i for i in range(1000, 2000)}
    large_category_map = {i: f'Category_{i}' for i in range(1000, 2000)}
    
    large_dataset = UserItemInteractionDataset(large_df, large_user_map, large_item_map, large_category_map, num_negatives=3)
    
    # Check if length is correct
    assert len(large_dataset) == 1000
    
    # Test retrieving a sample
    sample = large_dataset[500]
    assert len(sample["neg_items"]) <= 3  # Should have 3 or fewer negative items
    assert sample["pos_item"].item() not in sample["neg_items"].tolist()  # Should not contain the positive item


def test_out_of_bounds_negative_sampling():
    df_empty_negatives = pd.DataFrame({
        'client_id': [1],
        'sku': [101]
    })
    
    empty_negatives_user_map = {1: 0}
    empty_negatives_item_map = {101: 0}
    empty_negatives_category_map = {101: 'A'}
    
    empty_negatives_dataset = UserItemInteractionDataset(df_empty_negatives, empty_negatives_user_map, empty_negatives_item_map, empty_negatives_category_map, num_negatives=10)
    
    sample = empty_negatives_dataset[0]  # This should return a sample with 0 negative items, not an error
    assert sample["neg_items"].numel() == 0  # Expect an empty tensor for neg_items since no negatives were found