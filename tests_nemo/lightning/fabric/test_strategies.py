# from unittest.mock import MagicMock, create_autospec, patch

# from nemo.lightning import DataConfig, FabricMegatronStrategy, ModelParallelConfig
# from nemo.lightning.fabric.strategies import _MegatronDataLoaderIterDataFetcher
# from torch.optim import SGD


# class TestFabricMegatronStrategy:
#     @patch('nemo.lightning.fabric.strategies._strategy_lib.process_dataloader')
#     @patch('nemo.lightning.fabric.strategies.CombinedLoader')
#     @patch('nemo.lightning.fabric.strategies._MegatronDataLoaderIterDataFetcher')
#     def test_process_dataloader(self, mock_data_fetcher, mock_combined_loader, mock_process_dataloader) -> None:
#         mock_dataloader = MagicMock()
#         mock_process_dataloader.return_value = mock_dataloader
#         mock_combined_loader.return_value = mock_dataloader

#         # Create a mock object with the same interface as _MegatronDataLoaderIterDataFetcher
#         mock_data_fetcher_instance = create_autospec(_MegatronDataLoaderIterDataFetcher)
#         mock_data_fetcher.return_value = mock_data_fetcher_instance

#         strategy = FabricMegatronStrategy(
#             ModelParallelConfig(tensor_model_parallel_size=2),
#             DataConfig(256)
#         )
#         strategy.process_dataloader(mock_dataloader)

#         mock_process_dataloader.assert_called_once_with(mock_dataloader, strategy.data_config)
#         mock_combined_loader.assert_called_once_with(mock_dataloader, "max_size_cycle")
#         mock_data_fetcher.assert_called_once_with(strategy.data_config, output_data_idx=strategy.output_data_idx)
        
#     def test_setup_optimizer(self):
#         # Create a mock optimizer
#         mock_optimizer = MagicMock(spec=SGD)
#         mock_optimizer.on_megatron_step_start = MagicMock()

#         strategy = FabricMegatronStrategy(
#             ModelParallelConfig(tensor_model_parallel_size=2),
#             DataConfig(256)
#         )

#         strategy.setup_optimizer(mock_optimizer)
        
#         assert mock_optimizer in strategy.megatron_callbacks
