namespace MGroup.MachineLearning.Tests
{
	using System;
	using System.Collections.Generic;
	using System.Diagnostics;
	using System.Drawing;
	using System.Linq;
	using System.Reflection;
	using MGroup.MachineLearning;
	using MGroup.MachineLearning.Preprocessing;

	using Xunit;

	public class PrincipalComponentAnalysisTest
	{
		// PCA algorithm parameters
		private bool sort = true;
		private bool inPlace = false;
		private bool scaled = false;
		private int numberOfEigenvectors = 2;
		private double[,] dataSet = { { 1, 5 }, { 6, 2}, { 0.01, -0.02} };


		[Fact]
		private void TestDMAPAlgorithm()
		{
			PrincipalComponentAnalysis PCA = new PrincipalComponentAnalysis(dataSet, numberOfEigenvectors, sort, inPlace, scaled);
			PCA.ProcessData();
			Assert.True(Math.Abs(PCA.PCAeigenvalues[0] - 7.1038) < 0.001);
			Assert.True(Math.Abs(PCA.PCAeigenvalues[1] - 3.9416) < 0.001);
			Assert.True(Math.Abs(Math.Abs(PCA.PCAeigenvectors[0, 0]) - 0.5473) < 0.001);
			Assert.True(Math.Abs(Math.Abs(PCA.PCAeigenvectors[1, 0]) - 0.8369) < 0.001);
			Assert.True(Math.Abs(Math.Abs(PCA.PCAeigenvectors[2, 0]) - 0.0006) < 0.001);
			Assert.True(Math.Abs(Math.Abs(PCA.PCAeigenvectors[0, 1]) - 0.8369) < 0.001);
			Assert.True(Math.Abs(Math.Abs(PCA.PCAeigenvectors[1, 1]) - 0.5473) < 0.001);
			Assert.True(Math.Abs(Math.Abs(PCA.PCAeigenvectors[2, 1]) - 0.0056) < 0.001);
		}

	}
}
