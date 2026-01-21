from typing import List, Optional

class FixedLengthContextProcessor:
    """
    A processor for contexts where the context length exactly matches the chunk length.

    This class handles scenarios with perfect 1:1 mapping between context and chunk,
    ensuring no information loss and simplified processing.
    """

    @staticmethod
    def process_context(context: str, chunk_length: int = 12) -> List[str]:
        """
        Process a context with a fixed-length chunk strategy.

        Args:
            context (str): The input context to be processed
            chunk_length (int, optional): Length of each chunk. Defaults to 12.

        Returns:
            List[str]: A list containing the entire context as a single chunk

        Raises:
            ValueError: If context length does not match chunk length
        """
        if len(context) != chunk_length:
            raise ValueError(
                f"Context length must exactly match chunk length. "
                f"Context length: {len(context)}, Chunk length: {chunk_length}"
            )

        return [context]

    @staticmethod
    def process_task(context: str, chunk_length: int = 12) -> Optional[str]:
        """
        Perform a sample processing task on the context.

        Args:
            context (str): The input context to be processed
            chunk_length (int, optional): Length of each chunk. Defaults to 12.

        Returns:
            Optional[str]: Processed result or None if processing is not applicable
        """
        try:
            chunks = FixedLengthContextProcessor.process_context(context, chunk_length)
            return chunks[0]  # Since there's only one chunk
        except ValueError as e:
            print(f"Processing error: {e}")
            return None

# Example usage and demonstration
if __name__ == "__main__":
    # Perfect 12-character context scenario
    example_context = "Hello World!"
    result = FixedLengthContextProcessor.process_task(example_context)
    print(f"Processed Context: {result}")

    # Demonstrating error case
    try:
        invalid_context = "Too Long Context"  # More than 12 characters
        FixedLengthContextProcessor.process_task(invalid_context)
    except ValueError as e:
        print(f"Expected error: {e}")