namespace MGroup.MachineLearning
{
	using System;
	using System.Collections.Generic;
	using System.IO;
	using System.Linq;
	using System.Text;
	using MGroup.MachineLearning.Preprocessing;
	using Accord.Math;

	public class PrincipalComponentAnalysis
	{
		private double[,] dataSet;
		private int numberOfEigenvectors;
		private bool sort;
		private bool inPlace;
		private bool scaled;
		public double[] PCAeigenvalues;
		public double[,] PCAeigenvectors;
		public double[,] PCAeigenvectorsUntransformed;

		public PrincipalComponentAnalysis(double[,] dataSet, int numberOfEigenvectors, bool sort, bool inPlace, bool scaled)
		{
			this.dataSet = dataSet;
			this.numberOfEigenvectors = numberOfEigenvectors;
			this.sort = sort;
			this.inPlace = inPlace;
			this.scaled = scaled;
		}

		public void ProcessData()
		{
			int dimension1 = dataSet.GetLength(0);
			int dimension2 = dataSet.GetLength(1);

			if (dimension1 <= dimension2)
			{
				double[,] symmetricDataset = new double[dimension1, dimension1];
				for (var i = 0; i < dimension1; i++)
				{
					for (var j = 0; j < dimension1; j++)
					{
						for (var k = 0; k < dimension2; k++)
						{
							symmetricDataset[i, j] +=  dataSet[i, k] * dataSet[j, k];
						}
					}
				}
				(PCAeigenvalues, PCAeigenvectors) = EigenDecomposition.FindEigenValuesAndEigenvectorsSymmetricOnly(symmetricDataset, numberOfEigenvectors, inPlace, sort, scaled);
				for (var i = 0; i < PCAeigenvalues.Length; i++)
				{
					PCAeigenvalues[i] = Math.Pow(PCAeigenvalues[i], 0.5);
				}
			}
			else
			{
				double[,] symmetricDataset = new double[dimension2, dimension2];
				for (var i = 0; i < dimension2; i++)
				{
					for (var j = 0; j < dimension2; j++)
					{
						for (var k = 0; k < dimension1; k++)
						{
							symmetricDataset[i, j] +=  dataSet[k, i] * dataSet[k, j];
						}
					}
				};
				(PCAeigenvalues, PCAeigenvectorsUntransformed) = EigenDecomposition.FindEigenValuesAndEigenvectorsSymmetricOnly(symmetricDataset, numberOfEigenvectors, inPlace, sort, scaled);
				
				for (var i = 0; i < PCAeigenvalues.Length; i++)
				{
					PCAeigenvalues[i] = Math.Pow(PCAeigenvalues[i], 0.5);
				}
				double[,] tempMatrix = new double[dimension1, numberOfEigenvectors];
				for (var i = 0; i < dimension1; i++)
				{
					for (var j = 0; j < numberOfEigenvectors; j++)
					{
						for (var k = 0; k < dimension2; k++) 
						{
							tempMatrix[i, j] += dataSet[i, k] * PCAeigenvectorsUntransformed[k, j];
						}
					}
				}

				double[,] matrixOfEigenvalues = new double[numberOfEigenvectors, numberOfEigenvectors];
				for (var i = 0; i < numberOfEigenvectors; i++)
				{
					matrixOfEigenvalues[i, i] = 1/PCAeigenvalues[i];
				}

				PCAeigenvectors = new double[dimension1, numberOfEigenvectors];
				for (var i = 0; i < dimension1; i++)
				{
					for (var j = 0; j < numberOfEigenvectors; j++)
					{
						for (var k = 0; k < numberOfEigenvectors; k++)
						{
							PCAeigenvectors[i, j] += tempMatrix[i, j] * matrixOfEigenvalues[k, j];
						}
					}
				}
			}

		}
	}
}
