#include "gtest/gtest.h"

#include <time.h>
#include <stdlib.h>

#include "psrdada_cpp/effelsberg/edd/VLBI.cuh"

#include "psrdada_cpp/cuda_utils.hpp"
#include "thrust/extrema.h"

using namespace psrdada_cpp::effelsberg::edd;


TEST(BitManipulation, bitrange)
{
	uint32_t data = 0;
	EXPECT_THROW(setBitsWithValue(data, 0, 1, 4), std::runtime_error);
	EXPECT_NO_THROW(setBitsWithValue(data, 0, 2, 4));
}


TEST(BitManipulation, setBitsWithValue)
{
	uint32_t data = 0;
	setBitsWithValue(data, 0, 2, 3);
	EXPECT_EQ(data, 3);

	setBitsWithValue(data, 2, 4, 3);
	EXPECT_EQ(data, 15);

	EXPECT_EQ(3, getBitsValue(data, 1, 2));
	EXPECT_EQ(1, getBitsValue(data, 3, 12));

}

TEST(VDIFHeader, getSetWord0)
{
	VDIFHeader header;
	EXPECT_TRUE(header.isValid());

	header.setInvalid();
	EXPECT_FALSE(header.isValid());

	header.setValid();
	EXPECT_TRUE(header.isValid());

	EXPECT_EQ(header.getSecondsFromReferenceEpoch(), 0);
	header.setSecondsFromReferenceEpoch(12345);
	EXPECT_EQ(header.getSecondsFromReferenceEpoch(), 12345);
}

TEST(VDIFHeader, getSetWord1)
{
	VDIFHeader header;

	EXPECT_EQ(header.getReferenceEpoch(), 0);
	header.setReferenceEpoch(16);
	EXPECT_EQ(header.getReferenceEpoch(), 16);

	EXPECT_EQ(header.getDataFrameNumber(), 0);
	header.setDataFrameNumber(5);
	EXPECT_EQ(header.getDataFrameNumber(), 5);
}

TEST(VDIFHeader, getSetWord2)
{
	VDIFHeader header;
	EXPECT_EQ(header.getVersionNumber(), 1);

	EXPECT_EQ(header.getDataFrameLength(), 0);
	header.setDataFrameLength(1024);
	EXPECT_EQ(header.getDataFrameLength(), 1024);

	EXPECT_EQ(header.getNumberOfChannels(), 0);
	header.setNumberOfChannels(10);
	EXPECT_EQ(header.getNumberOfChannels(), 10);

}

TEST(VDIFHeader, getSetWord3)
{
	VDIFHeader header;
	EXPECT_TRUE(header.isRealDataType());
	EXPECT_FALSE(header.isComplexDataType());
	header.setComplexDataType();
	EXPECT_FALSE(header.isRealDataType());
	EXPECT_TRUE(header.isComplexDataType());

	EXPECT_EQ(header.getBitsPerSample(), 0);
	header.setBitsPerSample(13);
	EXPECT_EQ(header.getBitsPerSample(), 13);

	EXPECT_EQ(header.getThreadId(), 0);
	header.setThreadId(23);
	EXPECT_EQ(header.getThreadId(), 23);

	EXPECT_EQ(header.getStationId(), 0);
	header.setStationId(42);
	EXPECT_EQ(header.getStationId(), 42);
}



//int main(int argc, char **argv) {
//  ::testing::InitGoogleTest(&argc, argv);
//
//  return RUN_ALL_TESTS();
//}
