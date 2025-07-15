package wrappednormal;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.BeforeEach;
import static org.junit.jupiter.api.Assertions.*;

/**
 * Unit tests for WrappedBivariateNormal class.
 */
public class WrappedBivariateNormalTest {
    
    private WrappedBivariateNormal wbn;
    private static final double EPSILON = 1e-6;
    
    @BeforeEach
    public void setUp() {
        // Create a standard wrapped bivariate normal for testing
        wbn = new WrappedBivariateNormal(Math.PI, Math.PI/2, 0.5, 0.7, 0.3);
    }
    
    @Test
    public void testConstructorValidation() {
        // Test negative standard deviations
        assertThrows(IllegalArgumentException.class, () -> {
            new WrappedBivariateNormal(0, 0, -1, 1, 0);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            new WrappedBivariateNormal(0, 0, 1, -1, 0);
        });
        
        // Test correlation coefficient out of range
        assertThrows(IllegalArgumentException.class, () -> {
            new WrappedBivariateNormal(0, 0, 1, 1, 1.5);
        });
        
        assertThrows(IllegalArgumentException.class, () -> {
            new WrappedBivariateNormal(0, 0, 1, 1, -1.5);
        });
    }
    
    @Test
    public void testMeanWrapping() {
        // Test that means are wrapped to [0, 2π)
        WrappedBivariateNormal wbn1 = new WrappedBivariateNormal(3 * Math.PI, -Math.PI/2, 1, 1, 0);
        assertEquals(Math.PI, wbn1.getMu1(), EPSILON);
        assertEquals(3 * Math.PI / 2, wbn1.getMu2(), EPSILON);
    }
    
    @Test
    public void testPdfProperties() {
        // PDF should be non-negative
        for (double theta1 = 0; theta1 < 2 * Math.PI; theta1 += Math.PI / 4) {
            for (double theta2 = 0; theta2 < 2 * Math.PI; theta2 += Math.PI / 4) {
                assertTrue(wbn.pdf(theta1, theta2) >= 0);
            }
        }
        
        // PDF should be highest near the mean
        double pdfAtMean = wbn.pdf(wbn.getMu1(), wbn.getMu2());
        double pdfAway = wbn.pdf(0, 0);
        assertTrue(pdfAtMean > pdfAway);
    }
    
    @Test
    public void testPdfNormalization() {
        // Approximate integral of PDF over [0, 2π) × [0, 2π) should be close to 1
        double sum = 0.0;
        int n = 100;
        double delta = 2 * Math.PI / n;
        
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                double theta1 = i * delta;
                double theta2 = j * delta;
                sum += wbn.pdf(theta1, theta2) * delta * delta;
            }
        }
        
        assertEquals(1.0, sum, 0.01); // Allow 1% error due to discretization
    }
    
    @Test
    public void testSampling() {
        // Test single sample
        double[] sample = wbn.sample();
        assertEquals(2, sample.length);
        assertTrue(sample[0] >= 0 && sample[0] < 2 * Math.PI);
        assertTrue(sample[1] >= 0 && sample[1] < 2 * Math.PI);
        
        // Test multiple samples
        int n = 1000;
        double[][] samples = wbn.sample(n);
        assertEquals(n, samples.length);
        
        for (int i = 0; i < n; i++) {
            assertEquals(2, samples[i].length);
            assertTrue(samples[i][0] >= 0 && samples[i][0] < 2 * Math.PI);
            assertTrue(samples[i][1] >= 0 && samples[i][1] < 2 * Math.PI);
        }
    }
    
    @Test
    public void testSamplingWithSeed() {
        // Test reproducibility with seed
        long seed = 12345L;
        WrappedBivariateNormal wbn1 = new WrappedBivariateNormal(0, 0, 1, 1, 0.5, seed);
        WrappedBivariateNormal wbn2 = new WrappedBivariateNormal(0, 0, 1, 1, 0.5, seed);
        
        double[] sample1 = wbn1.sample();
        double[] sample2 = wbn2.sample();
        
        assertArrayEquals(sample1, sample2, EPSILON);
    }
    
    @Test
    public void testCircularStatistics() {
        // Circular variances should be between 0 and 1
        assertTrue(wbn.circularVariance1() >= 0 && wbn.circularVariance1() <= 1);
        assertTrue(wbn.circularVariance2() >= 0 && wbn.circularVariance2() <= 1);
        
        // Test extreme cases
        // Small sigma -> small circular variance
        WrappedBivariateNormal concentrated = new WrappedBivariateNormal(0, 0, 0.1, 0.1, 0);
        assertTrue(concentrated.circularVariance1() < 0.01);
        
        // Large sigma -> circular variance approaching 1
        WrappedBivariateNormal dispersed = new WrappedBivariateNormal(0, 0, 5, 5, 0);
        assertTrue(dispersed.circularVariance1() > 0.99);
    }
    
    @Test
    public void testLogPdf() {
        // log(pdf) should equal log of pdf
        double theta1 = Math.PI / 3;
        double theta2 = 2 * Math.PI / 3;
        
        double pdf = wbn.pdf(theta1, theta2);
        double logPdf = wbn.logPdf(theta1, theta2);
        
        assertEquals(Math.log(pdf), logPdf, EPSILON);
    }
    
    @Test
    public void testSymmetry() {
        // Test symmetry for uncorrelated case
        WrappedBivariateNormal symmetric = new WrappedBivariateNormal(Math.PI, Math.PI, 1, 1, 0);
        
        double pdf1 = symmetric.pdf(Math.PI + 0.5, Math.PI - 0.5);
        double pdf2 = symmetric.pdf(Math.PI - 0.5, Math.PI + 0.5);
        
        assertEquals(pdf1, pdf2, EPSILON);
    }
    
    @Test
    public void testGetters() {
        assertEquals(Math.PI, wbn.getMu1(), EPSILON);
        assertEquals(Math.PI / 2, wbn.getMu2(), EPSILON);
        assertEquals(0.5, wbn.getSigma1(), EPSILON);
        assertEquals(0.7, wbn.getSigma2(), EPSILON);
        assertEquals(0.3, wbn.getRho(), EPSILON);
    }
    
    @Test
    public void testToString() {
        String str = wbn.toString();
        assertTrue(str.contains("WrappedBivariateNormal"));
        assertTrue(str.contains("mu1="));
        assertTrue(str.contains("mu2="));
        assertTrue(str.contains("sigma1="));
        assertTrue(str.contains("sigma2="));
        assertTrue(str.contains("rho="));
    }
}
