"""
Sorting Assignment Starter Code
Implement five sorting algorithms and benchmark their performance.
"""

import json
import time
import random
import tracemalloc


# ============================================================================
# PART 1: SORTING IMPLEMENTATIONS
# ============================================================================

def bubble_sort(arr):
    """
    Sort array using bubble sort algorithm.
    
    Bubble sort repeatedly steps through the list, compares adjacent elements,
    and swaps them if they're in the wrong order.
    
    Args:
        arr (list): List of integers to sort
    
    Returns:
        list: Sorted list in ascending order
    
    Example:
        bubble_sort([64, 34, 25, 12, 22, 11, 90]) returns [11, 12, 22, 25, 34, 64, 90]
    """
    # TODO: Implement bubble sort
    # Hint: Use nested loops - outer loop for passes, inner loop for comparisons
    # Hint: Compare adjacent elements and swap if left > right
    
    sorted_arr = arr.copy()
    size = len(sorted_arr)

    for i in range(size): #loop to access each element
        for j in range(0, size - i - 1): #loop to compare array elements
            if sorted_arr[j] > sorted_arr[j+1]: #compare two adjacent elems
                temp = sorted_arr[j] 
                sorted_arr[j] = sorted_arr[j+1]
                sorted_arr[j+1]= temp
    return sorted_arr  # Return the sorted array

print(bubble_sort([64, 34, 25, 12, 22, 11, 90])) #test case


def selection_sort(arr):
    """
    Sort array using selection sort algorithm.
    
    Selection sort divides the list into sorted and unsorted regions, repeatedly
    selecting the minimum element from unsorted region and moving it to sorted region.
    
    Args:
        arr (list): List of integers to sort
    
    Returns:
        list: Sorted list in ascending order
    
    Example:
        selection_sort([64, 34, 25, 12, 22, 11, 90]) returns [11, 12, 22, 25, 34, 64, 90]
    """
    # TODO: Implement selection sort
    # Hint: Find minimum element in unsorted portion, swap it with first unsorted element
    
    sorted_arr = arr.copy()
    size = len(sorted_arr)

    for step in range(size):
        min_idx = step

        for i in range(step + 1, size): #select min elem in each loop
            if sorted_arr[i] < sorted_arr[min_idx]: 
                min_idx = i

        temp = sorted_arr[step]
        sorted_arr[step] = sorted_arr[min_idx] #min in correct position
        sorted_arr[min_idx] = temp

    return sorted_arr  # Return the sorted array
print(selection_sort([64, 34, 25, 12, 22, 11, 90])) #test case


def insertion_sort(arr):
    """
    Sort array using insertion sort algorithm.
    
    Insertion sort builds the final sorted array one item at a time, inserting
    each element into its proper position in the already-sorted portion.
    
    Args:
        arr (list): List of integers to sort
    
    Returns:
        list: Sorted list in ascending order
    
    Example:
        insertion_sort([64, 34, 25, 12, 22, 11, 90]) returns [11, 12, 22, 25, 34, 64, 90]
    """
    # TODO: Implement insertion sort
    # Hint: Start from second element, insert it into correct position in sorted portion
    
    sorted_arr = arr.copy()

    for step in range(1,len(sorted_arr)):
        key = sorted_arr[step]
        j = step - 1

        while j>= 0 and key < sorted_arr[j]: #compare key with each elem on left of it
            sorted_arr[j+1] = sorted_arr[j]
            j = j - 1
        #place key at after the elem just smaller than it
        sorted_arr[j+1] = key 

    return sorted_arr  # Return the sorted array
print(insertion_sort([64, 34, 25, 12, 22, 11, 90])) #test case

def merge_sort(arr):
    """
    Sort array using merge sort algorithm.
    
    Merge sort is a divide-and-conquer algorithm that divides the array into halves,
    recursively sorts them, and then merges the sorted halves.
    
    Args:
        arr (list): List of integers to sort
    
    Returns:
        list: Sorted list in ascending order
    
    Example:
        merge_sort([64, 34, 25, 12, 22, 11, 90]) returns [11, 12, 22, 25, 34, 64, 90]
    """
    # TODO: Implement merge sort
    # Hint: Base case - if array has 1 or 0 elements, it's already sorted
    # Hint: Recursive case - split array in half, sort each half, merge sorted halves
    # Hint: You'll need a helper function to merge two sorted arrays
    
    sorted_arr = arr.copy()
    merge_sort_helper(sorted_arr)
    return sorted_arr

def merge_sort_helper(arr): #I asked chatgpt to help explain this portion to me because I was a bit confused about the structure of the code and programiz's explanation.
    if len(arr) > 1:
        mid = len(arr) // 2 # Finding the mid of the array
        L = arr[:mid] # Dividing the array elements into 2 halves
        R = arr[mid:]

        merge_sort_helper(L) # Sorting the first half
        merge_sort_helper(R) # Sorting the second half

        i = j = k = 0
        
        while i < len(L) and j < len(R):
            if L[i] < R[j]:
                arr[k] = L[i]
                i += 1
            else:
                arr[k] = R[j]
                j += 1
            k += 1

        while i < len(L):
            arr[k] = L[i]
            i += 1
            k += 1

        while j < len(R):
            arr[k] = R[j]
            j += 1
            k += 1

print(merge_sort([64, 34, 25, 12, 22, 11, 90])) #test case



# ============================================================================
# PART 2: STABILITY DEMONSTRATION
# ============================================================================

def demonstrate_stability():
    """
    Demonstrate which sorting algorithms are stable by sorting products by price.
    
    Creates a list of product dictionaries with prices and original order.
    Sorts by price and checks if products with same price maintain original order.
    
    Returns:
        dict: Results showing which algorithms preserved order for equal elements
    """

    # Sample products with duplicate prices
    products = [
        {"name": "Widget A", "price": 1999, "original_position": 0},
        {"name": "Gadget B", "price": 999, "original_position": 1},
        {"name": "Widget C", "price": 1999, "original_position": 2},
        {"name": "Tool D", "price": 999, "original_position": 3},
        {"name": "Widget E", "price": 1999, "original_position": 4},
    ]

    algorithms = {
        "bubble_sort": bubble_sort,
        "selection_sort": selection_sort,
        "insertion_sort": insertion_sort,
        "merge_sort": merge_sort
    }

    results = {}

    # Test each algorithm
    for algo_name, algo_func in algorithms.items():

        # Extract just the prices
        prices = [product["price"] for product in products]

        # Sort prices using the algorithm
        sorted_prices = algo_func(prices.copy())

        # Rebuild sorted product list based on sorted prices
        sorted_products = []
        used_indices = set()

        for sorted_price in sorted_prices:
            for i, product in enumerate(products):
                # Match price AND make sure we don't reuse same item
                if product["price"] == sorted_price and i not in used_indices:
                    sorted_products.append(product)
                    used_indices.add(i)
                    break

        stable = True

        # Check price 999 group
        group_999 = [p for p in sorted_products if p["price"] == 999]
        if len(group_999) >= 2:
            if group_999[0]["original_position"] > group_999[1]["original_position"]:
                stable = False

        # Check price 1999 group
        group_1999 = [p for p in sorted_products if p["price"] == 1999]
        for i in range(len(group_1999) - 1):
            if group_1999[i]["original_position"] > group_1999[i + 1]["original_position"]:
                stable = False

        results[algo_name] = "stable" if stable else "unstable"

    return results

# ============================================================================
# PART 3: PERFORMANCE BENCHMARKING
# ============================================================================

def load_dataset(filename):
    """Load a dataset from JSON file."""
    with open(f"datasets/{filename}", "r") as f:
        return json.load(f)


def load_test_cases():
    """Load test cases for validation."""
    with open("datasets/test_cases.json", "r") as f:
        return json.load(f)


def test_sorting_correctness():
    """Test that sorting functions work correctly on small test cases."""
    print("="*70)
    print("TESTING SORTING CORRECTNESS")
    print("="*70 + "\n")
    
    test_cases = load_test_cases()
    
    test_names = ["small_random", "small_sorted", "small_reverse", "small_duplicates"]
    algorithms = {
        "Bubble Sort": bubble_sort,
        "Selection Sort": selection_sort,
        "Insertion Sort": insertion_sort,
        "Merge Sort": merge_sort
    }
    
    for test_name in test_names:
        print(f"Test: {test_name}")
        print(f"  Input:    {test_cases[test_name]}")
        print(f"  Expected: {test_cases['expected_sorted'][test_name]}")
        print()
        
        for algo_name, algo_func in algorithms.items():
            try:
                result = algo_func(test_cases[test_name].copy())
                expected = test_cases['expected_sorted'][test_name]
                status = " PASS" if result == expected else " FAIL"
                print(f"    {algo_name:20s}: {result} {status}")
            except Exception as e:
                print(f"    {algo_name:20s}: ERROR - {str(e)}")
        
        print()


def benchmark_algorithm(sort_func, data):
    """
    Benchmark a sorting algorithm on given data.
    
    Args:
        sort_func: The sorting function to test
        data: The dataset to sort (will be copied so original isn't modified)
    
    Returns:
        tuple: (execution_time_ms, peak_memory_kb)
    """
    # Copy data so we don't modify original
    data_copy = data.copy()
    
    # Start memory tracking
    tracemalloc.start()
    
    # Measure execution time
    start_time = time.perf_counter()
    sort_func(data_copy)
    end_time = time.perf_counter()
    
    # Get peak memory usage
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    execution_time_ms = (end_time - start_time) * 1000
    peak_memory_kb = peak / 1024
    
    return execution_time_ms, peak_memory_kb


def benchmark_all_datasets():
    """Benchmark all sorting algorithms on all datasets."""
    print("\n" + "="*70)
    print("BENCHMARKING SORTING ALGORITHMS")
    print("="*70 + "\n")
    
    datasets = {
        "orders.json": ("Order Processing Queue", 50000, 5000),
        "products.json": ("Product Catalog", 100000, 5000),
        "inventory.json": ("Inventory Reconciliation", 25000, 5000),
        "activity_log.json": ("Customer Activity Log", 75000, 5000)
    }
    
    algorithms = {
        "Bubble Sort": bubble_sort,
        "Selection Sort": selection_sort,
        "Insertion Sort": insertion_sort,
        "Merge Sort": merge_sort
    }
    
    for filename, (description, full_size, sample_size) in datasets.items():
        print(f"Dataset: {description} ({sample_size:,} element sample)")
        print("-" * 70)
        
        data = load_dataset(filename)
        # Use first sample_size elements for fair comparison
        data_sample = data[:sample_size]
        
        for algo_name, algo_func in algorithms.items():
            try:
                exec_time, memory = benchmark_algorithm(algo_func, data_sample)
                print(f"  {algo_name:20s}: {exec_time:8.2f} ms | {memory:8.2f} KB")
            except Exception as e:
                print(f"  {algo_name:20s}: ERROR - {str(e)}")
        
        print()


def analyze_stability():
    """Test and display which algorithms are stable."""
    print("="*70)
    print("STABILITY ANALYSIS")
    print("="*70 + "\n")
    
    print("Testing which algorithms preserve order of equal elements...\n")
    
    results = demonstrate_stability()
    
    for algo_name, stability in results.items():
        print(f"  {algo_name:20s}: {stability}")
    
    print()


if __name__ == "__main__":
    print("SORTING ASSIGNMENT - STARTER CODE")
    print("Implement the sorting functions above, then run tests.\n")
    
    # Uncomment these as you complete each part:
    
    test_sorting_correctness()
    benchmark_all_datasets()
    analyze_stability()
    
    print("\n Uncomment the test functions in the main block to run benchmarks!")