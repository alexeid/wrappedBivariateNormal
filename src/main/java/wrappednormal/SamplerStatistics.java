package wrappednormal;

/**
 * Statistics for evaluating sampler performance.
 */
public class SamplerStatistics {
    private long totalSamples;
    private long totalTime;
    private long totalAttempts;
    private double acceptanceRate;
    private double samplesPerSecond;

    public SamplerStatistics() {
        reset();
    }

    public void reset() {
        totalSamples = 0;
        totalTime = 0;
        totalAttempts = 0;
        acceptanceRate = 0;
        samplesPerSecond = 0;
    }

    public void recordSample(long timeTaken) {
        totalSamples++;
        totalTime += timeTaken;
        updateRates();
    }

    public void recordAttempt() {
        totalAttempts++;
    }

    public void recordBatch(int samples, long timeTaken, int attempts) {
        totalSamples += samples;
        totalTime += timeTaken;
        totalAttempts += attempts;
        updateRates();
    }

    private void updateRates() {
        if (totalAttempts > 0) {
            acceptanceRate = (double) totalSamples / totalAttempts;
        }
        if (totalTime > 0) {
            samplesPerSecond = (double) totalSamples * 1000 / totalTime;
        }
    }

    // Getters
    public long getTotalSamples() { return totalSamples; }
    public long getTotalTime() { return totalTime; }
    public long getTotalAttempts() { return totalAttempts; }
    public double getAcceptanceRate() { return acceptanceRate; }
    public double getSamplesPerSecond() { return samplesPerSecond; }

    @Override
    public String toString() {
        return String.format("Samples: %d, Time: %d ms, Attempts: %d, " +
                        "Acceptance: %.2f%%, Rate: %.1f samples/sec",
                totalSamples, totalTime, totalAttempts,
                acceptanceRate * 100, samplesPerSecond);
    }
}