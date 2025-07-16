package wrappednormal;

/**
 * Interface for sampling from a product of wrapped bivariate normal distributions.
 * Different implementations can use different sampling strategies.
 */
public interface ProductSampler {

    /**
     * Generates a single sample from the product distribution.
     *
     * @return A 2-element array containing the sampled angles in [0, 2π)
     */
    double[] sample();

    /**
     * Generates multiple samples from the product distribution.
     *
     * @param n Number of samples to generate
     * @return An n×2 array of samples
     */
    double[][] sample(int n);

    /**
     * Gets the name of the sampling method.
     *
     * @return Name of the sampler
     */
    String getName();

    /**
     * Gets a description of the sampling method.
     *
     * @return Description of how the sampler works
     */
    String getDescription();

    /**
     * Resets any internal state of the sampler.
     */
    void reset();

    /**
     * Gets performance statistics for the sampler.
     *
     * @return Statistics about the sampler's performance
     */
    SamplerStatistics getStatistics();

    /**
     * Sets the random seed for reproducible sampling.
     *
     * @param seed Random seed
     */
    void setSeed(long seed);
}