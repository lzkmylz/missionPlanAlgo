"""
Fragmented storage model for satellite on-board storage simulation.

Simulates real satellite solid-state storage with filesystem overhead,
block-based allocation, and fragmentation tracking.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


@dataclass
class StorageBlock:
    """Storage block (simulates filesystem block)

    Attributes:
        start_address: Block start address in bytes
        size: Block size in bytes
        is_allocated: Whether block is allocated
        file_id: ID of file occupying block (if allocated)
        created_time: Block creation time
        last_accessed: Last access time
    """
    start_address: int
    size: int
    is_allocated: bool = False
    file_id: Optional[str] = None
    created_time: datetime = None
    last_accessed: datetime = None

    def __post_init__(self):
        """Initialize timestamps if not provided"""
        if self.created_time is None:
            self.created_time = datetime.now()
        if self.last_accessed is None:
            self.last_accessed = datetime.now()


class FragmentedStorageModel:
    """Fragmented storage model simulating real satellite solid-state storage

    Models filesystem-level behavior including:
    - Block-based allocation (like SSDs)
    - Filesystem overhead (reserved space)
    - File metadata overhead
    - Fragmentation due to allocation/deallocation patterns

    Example:
        storage = FragmentedStorageModel(
            total_capacity=500*1024*1024*1024,  # 500GB
            block_size=4096,
            filesystem_overhead=0.15
        )

        # Allocate space for imaging data
        success, allocated = storage.allocate('IMG_001', 1024*1024*1024)
        if success:
            print(f"Allocated {allocated} bytes")

        # Check fragmentation status
        status = storage.get_storage_status()
        print(f"Fragmentation: {status['fragmentation_ratio']:.2%}")

    Attributes:
        total_capacity: Total storage capacity in bytes
        block_size: Filesystem block size in bytes
        filesystem_overhead: Fraction reserved for filesystem (0-1)
        effective_capacity: User-accessible capacity in bytes
    """

    def __init__(self,
                 total_capacity: int,
                 block_size: int = 4096,
                 filesystem_overhead: float = 0.15,
                 metadata_size: int = 256):
        """Initialize fragmented storage model

        Args:
            total_capacity: Total storage capacity (bytes)
            block_size: Filesystem block size (bytes)
            filesystem_overhead: Filesystem reserved fraction (0-1)
            metadata_size: Per-file metadata size (bytes)
        """
        self.total_capacity = total_capacity
        self.block_size = block_size
        self.metadata_size = metadata_size
        self.filesystem_overhead = filesystem_overhead

        # Effective capacity after overhead
        self.effective_capacity = int(total_capacity * (1 - filesystem_overhead))

        # Initialize blocks
        num_blocks = self.effective_capacity // block_size
        self.blocks: List[StorageBlock] = [
            StorageBlock(
                start_address=i * block_size,
                size=block_size,
                is_allocated=False
            )
            for i in range(num_blocks)
        ]

        # File index: file_id -> list of block indices
        self.files: Dict[str, List[int]] = {}

        logger.info(
            f"Initialized storage: {self.total_capacity / (1024**3):.2f} GB total, "
            f"{self.effective_capacity / (1024**3):.2f} GB effective, "
            f"{num_blocks} blocks"
        )

    @property
    def used_space(self) -> int:
        """Used space including metadata (bytes)"""
        used_blocks = sum(1 for block in self.blocks if block.is_allocated)
        metadata_overhead = len(self.files) * self.metadata_size
        return used_blocks * self.block_size + metadata_overhead

    @property
    def free_space(self) -> int:
        """Free space (bytes)"""
        return self.effective_capacity - self.used_space

    @property
    def fragmentation_ratio(self) -> float:
        """Fragmentation ratio (0-1)

        Defined as proportion of free blocks that are not contiguous.
        Higher values indicate more fragmented storage.
        """
        free_blocks = [b for b in self.blocks if not b.is_allocated]
        if not free_blocks:
            return 0.0

        # Count fragmented free blocks
        fragmented_free = 0
        for i in range(len(free_blocks) - 1):
            current_end = free_blocks[i].start_address + free_blocks[i].size
            next_start = free_blocks[i + 1].start_address
            if current_end != next_start:
                fragmented_free += 1

        return fragmented_free / len(free_blocks)

    def allocate(self, file_id: str, size: int) -> Tuple[bool, int]:
        """Allocate storage space using first-fit algorithm

        Args:
            file_id: Unique file identifier
            size: Requested size in bytes

        Returns:
            Tuple of (success, actual_allocated_size)
        """
        if file_id in self.files:
            logger.warning(f"File {file_id} already exists")
            return False, 0

        # Calculate required blocks
        num_blocks_needed = (size + self.block_size - 1) // self.block_size
        total_size = num_blocks_needed * self.block_size + self.metadata_size

        if self.used_space + total_size > self.effective_capacity:
            logger.warning(f"Insufficient space for {file_id}: need {total_size}, have {self.free_space}")
            return False, 0

        # Find contiguous free blocks (first-fit)
        allocated_blocks = []
        consecutive_free = 0
        start_idx = -1

        for i, block in enumerate(self.blocks):
            if not block.is_allocated:
                if consecutive_free == 0:
                    start_idx = i
                consecutive_free += 1

                if consecutive_free >= num_blocks_needed:
                    allocated_blocks = list(range(start_idx, start_idx + num_blocks_needed))
                    break
            else:
                consecutive_free = 0
                start_idx = -1

        # If no contiguous space, try fragmented allocation
        if len(allocated_blocks) < num_blocks_needed:
            free_indices = [i for i, b in enumerate(self.blocks) if not b.is_allocated]
            if len(free_indices) >= num_blocks_needed:
                allocated_blocks = free_indices[:num_blocks_needed]
            else:
                logger.warning(f"Not enough free blocks for {file_id}")
                return False, 0

        # Mark blocks as allocated
        now = datetime.now()
        for idx in allocated_blocks:
            self.blocks[idx].is_allocated = True
            self.blocks[idx].file_id = file_id
            self.blocks[idx].last_accessed = now

        self.files[file_id] = allocated_blocks

        actual_size = len(allocated_blocks) * self.block_size
        logger.debug(f"Allocated {actual_size} bytes for {file_id} in {len(allocated_blocks)} blocks")

        return True, actual_size

    def deallocate(self, file_id: str) -> bool:
        """Deallocate (delete) a file

        Args:
            file_id: File to delete

        Returns:
            True if successful
        """
        if file_id not in self.files:
            logger.warning(f"File {file_id} not found")
            return False

        # Free blocks
        for idx in self.files[file_id]:
            self.blocks[idx].is_allocated = False
            self.blocks[idx].file_id = None

        del self.files[file_id]
        logger.debug(f"Deallocated file {file_id}")

        return True

    def get_file_size(self, file_id: str) -> int:
        """Get size of allocated file

        Args:
            file_id: File identifier

        Returns:
            File size in bytes (0 if not found)
        """
        if file_id not in self.files:
            return 0

        return len(self.files[file_id]) * self.block_size

    def check_storage_feasibility(self, required_size: int) -> Tuple[bool, int]:
        """Check if storage operation is feasible

        Args:
            required_size: Required space in bytes

        Returns:
            Tuple of (is_feasible, available_space)
        """
        # Account for metadata overhead
        total_required = required_size + self.metadata_size

        # Check if we have enough space
        if self.free_space < total_required:
            return False, self.free_space

        # Check if we can find contiguous space (best case)
        num_blocks_needed = (required_size + self.block_size - 1) // self.block_size

        consecutive_free = 0
        for block in self.blocks:
            if not block.is_allocated:
                consecutive_free += 1
                if consecutive_free >= num_blocks_needed:
                    return True, self.free_space
            else:
                consecutive_free = 0

        # Can fit fragmented but not contiguous
        free_blocks = sum(1 for b in self.blocks if not b.is_allocated)
        if free_blocks >= num_blocks_needed:
            return True, self.free_space

        return False, self.free_space

    def get_storage_status(self) -> Dict[str, float]:
        """Get storage status report

        Returns:
            Dictionary with storage statistics
        """
        total_blocks = len(self.blocks)
        used_blocks = sum(1 for b in self.blocks if b.is_allocated)

        return {
            'total_capacity_gb': self.total_capacity / (1024**3),
            'effective_capacity_gb': self.effective_capacity / (1024**3),
            'used_space_gb': self.used_space / (1024**3),
            'free_space_gb': self.free_space / (1024**3),
            'utilization_ratio': self.used_space / self.effective_capacity if self.effective_capacity > 0 else 0,
            'fragmentation_ratio': self.fragmentation_ratio,
            'file_count': len(self.files),
            'total_blocks': total_blocks,
            'used_blocks': used_blocks,
            'free_blocks': total_blocks - used_blocks,
        }

    def defragment(self) -> int:
        """Defragment storage (compact free space)

        This is a simulated operation - in real systems this would
        involve moving allocated blocks to create larger contiguous regions.

        Returns:
            Number of blocks moved (simulated)
        """
        # This is a placeholder for a real defragmentation algorithm
        # In practice, this would reorder blocks to consolidate free space

        logger.info("Storage defragmentation requested (simulated)")

        # Calculate current fragmentation
        before_frag = self.fragmentation_ratio

        # Simulate defragmentation by recreating blocks in order
        allocated = [(i, b) for i, b in enumerate(self.blocks) if b.is_allocated]

        # Reset all blocks
        for block in self.blocks:
            block.is_allocated = False
            block.file_id = None

        # Re-allocate in order
        idx = 0
        for _, block in allocated:
            self.blocks[idx].is_allocated = True
            self.blocks[idx].file_id = block.file_id
            idx += 1

        after_frag = self.fragmentation_ratio
        logger.info(f"Defragmentation complete: {before_frag:.2%} -> {after_frag:.2%}")

        return len(allocated)
