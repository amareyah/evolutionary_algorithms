import pytest
import numpy as np
from sga import SGA


def objective_func(x):
    return x * np.sin(10 * np.pi * x) + 2.0


x_MIN = -1
x_MAX = 2
CHROMOSOME_SIZE = 12
POPULATION_SIZE = 20


@pytest.fixture
def default_SGA():
    """Returns SGA instance"""
    return SGA(
        objective_func,
        population_size=POPULATION_SIZE,
        chromo_size=CHROMOSOME_SIZE,
        x_min=x_MIN,
        x_max=x_MAX,
    )


@pytest.fixture
def large_SGA():
    """Returns SGA instance"""
    return SGA(
        objective_func, population_size=10000, chromo_size=16, x_min=x_MIN, x_max=x_MAX,
    )


def test_initialize_genotype_population(default_SGA):
    default_SGA.initialize_genotype_population()
    assert default_SGA.population.shape == (POPULATION_SIZE, CHROMOSOME_SIZE)
    assert default_SGA.population.dtype == bool


@pytest.fixture
def sample_genotype():
    x1 = np.random.randint(
        low=0, high=2, size=(1, POPULATION_SIZE, CHROMOSOME_SIZE), dtype=bool
    )
    x2 = np.vstack(([1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1],) * POPULATION_SIZE).astype(
        bool
    )

    x2 = np.expand_dims(x2, axis=0)
    x = np.concatenate((x1, x2), axis=0)
    return x


@pytest.mark.parametrize("a,b", [(x_MIN, x_MAX)])
def test_binary_to_float(default_SGA, sample_genotype, a, b):
    assert sample_genotype.shape == (2, POPULATION_SIZE, CHROMOSOME_SIZE)

    result = default_SGA.binary_to_float(sample_genotype, a, b)

    assert result.shape == (sample_genotype.shape[0], sample_genotype.shape[1])
    assert result[-1, -1] - 0.9533691406 < 1e-07


@pytest.mark.parametrize("a,b", [(x_MIN, x_MAX)])
def test_get_fittness_values(default_SGA, sample_genotype, a, b):
    result = default_SGA.get_fittenss_values(sample_genotype)
    assert result.shape == (sample_genotype.shape[0], sample_genotype.shape[1])
    assert np.all(
        result == objective_func(default_SGA.binary_to_float(sample_genotype, a, b))
    )


def test_get_selection_probs(default_SGA):
    with pytest.raises(Exception):
        some_vals = np.random.random_sample((30, 5)) * 5 + 1
        default_SGA.get_selection_probs(some_vals)

    fittness_vals = np.random.random_sample((30,)) * 5 + 1
    result = default_SGA.get_selection_probs(fittness_vals)

    assert np.abs(np.sum(result) - 1) < 1e-9


@pytest.mark.parametrize("dummy_var", list(range(5)))
def test_create_mating_pool(large_SGA, dummy_var):
    large_SGA.initialize_genotype_population()
    large_SGA.create_mating_pool()
    assert large_SGA.mating_pool_fittness.mean() > large_SGA.population_fittness.mean()


def test_single_point_crossover(default_SGA):
    np.random.seed(1)

    case_1 = [[0, 1, 0, 1, 1, 1, 0, 0], [1, 0, 0, 1, 0, 1, 1, 0]]

    result_1 = default_SGA.single_point_crossover(case_1)

    assert np.all(result_1 == [[1, 0, 0, 1, 0, 1, 0, 0], [0, 1, 0, 1, 1, 1, 1, 0]])

    case_2 = [[0, 0, 0, 1, 1, 0, 1, 0], [1, 0, 1, 1, 0, 0, 1, 1]]

    result_2 = default_SGA.single_point_crossover(case_2)

    assert np.all(result_2 == [[1, 0, 1, 1, 1, 0, 1, 0], [0, 0, 0, 1, 0, 0, 1, 1]])


def test_perform_variation_doesnot_change_old_population(default_SGA):
    offsprings = default_SGA.perform_variation()
    offspr_fittness = default_SGA.get_fittenss_values(offsprings)

    old_generation = default_SGA.mating_pool
    old_gener_fitness = default_SGA.get_fittenss_values(old_generation)

    assert np.all(old_gener_fitness == default_SGA.mating_pool_fittness)
    assert np.any(offspr_fittness != old_gener_fitness)


@pytest.mark.parametrize(
    "x, n", [(np.random.rand(6), 2), (np.random.rand(7), 2), (np.random.rand(7), 3)]
)
def test_split_to_chunks(default_SGA, x, n):
    if len(x) % n != 0:
        with pytest.raises(Exception) as e_info:
            default_SGA.split_to_chunks(x, n)
            assert (
                e_info.value.args[0]
                == "length of x should be divisible on n with remainder zero."
            )
    else:
        chunk_lengths = [len(ch) == n for ch in default_SGA.split_to_chunks(x, n)]
        assert np.all(chunk_lengths)
