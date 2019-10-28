#include "gtest/gtest.h"

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
	EXPECT_EQ(data, 3u);

	setBitsWithValue(data, 2, 4, 3);
	EXPECT_EQ(data, 15u);

	EXPECT_EQ(3u, getBitsValue(data, 1, 2));
	EXPECT_EQ(1u, getBitsValue(data, 3, 12));

}

TEST(VDIFHeader, getSetWord0)
{
	VDIFHeader header;
	EXPECT_TRUE(header.isValid());

	header.setInvalid();
	EXPECT_FALSE(header.isValid());

	header.setValid();
	EXPECT_TRUE(header.isValid());

	EXPECT_EQ(header.getSecondsFromReferenceEpoch(), 0u);
	header.setSecondsFromReferenceEpoch(12345);
	EXPECT_EQ(header.getSecondsFromReferenceEpoch(), 12345u);
}

TEST(VDIFHeader, getSetWord1)
{
	VDIFHeader header;

	EXPECT_EQ(header.getReferenceEpoch(), 0u);
	header.setReferenceEpoch(16);
	EXPECT_EQ(header.getReferenceEpoch(), 16u);

	EXPECT_EQ(header.getDataFrameNumber(), 0u);
	header.setDataFrameNumber(5);
	EXPECT_EQ(header.getDataFrameNumber(), 5u);
}

TEST(VDIFHeader, getSetWord2)
{
	VDIFHeader header;
	EXPECT_EQ(header.getVersionNumber(), 1u);

	EXPECT_EQ(header.getDataFrameLength(), 0u);
	header.setDataFrameLength(1024);
	EXPECT_EQ(header.getDataFrameLength(), 1024u);

	EXPECT_EQ(header.getNumberOfChannels(), 0u);
	header.setNumberOfChannels(10);
	EXPECT_EQ(header.getNumberOfChannels(), 10u);

}

TEST(VDIFHeader, getSetWord3)
{
	VDIFHeader header;
	EXPECT_TRUE(header.isRealDataType());
	EXPECT_FALSE(header.isComplexDataType());
	header.setComplexDataType();
	EXPECT_FALSE(header.isRealDataType());
	EXPECT_TRUE(header.isComplexDataType());

	EXPECT_EQ(header.getBitsPerSample(), 0u);
	header.setBitsPerSample(13);
	EXPECT_EQ(header.getBitsPerSample(), 13u);

	EXPECT_EQ(header.getThreadId(), 0u);
	header.setThreadId(23);
	EXPECT_EQ(header.getThreadId(), 23u);

	EXPECT_EQ(header.getStationId(), 0u);
	header.setStationId(42);
	EXPECT_EQ(header.getStationId(), 42u);
}

TEST(VDIFHeader, testTimeStampConversion)
{
	VDIFHeader header;
  size_t currentTime = 1554915838;
  header.setTimeReferencesFromTimestamp(currentTime);

  EXPECT_EQ(currentTime, header.getTimestamp()) << "Reference epoch: " << header.getReferenceEpoch() << " + " << header.getSecondsFromReferenceEpoch() << " s";

  header.setTimeReferencesFromTimestamp(946684800);
  EXPECT_EQ(0u, header.getReferenceEpoch());
  EXPECT_EQ(0u, header.getSecondsFromReferenceEpoch());

  header.setTimeReferencesFromTimestamp(962409600);
  EXPECT_EQ(1u, header.getReferenceEpoch());
  EXPECT_EQ(0u, header.getSecondsFromReferenceEpoch());

  header.setTimeReferencesFromTimestamp(962409600 + 100);
  EXPECT_EQ(1u, header.getReferenceEpoch());
  EXPECT_EQ(100u, header.getSecondsFromReferenceEpoch());
}

//int main(int argc, char **argv) {
//  ::testing::InitGoogleTest(&argc, argv);
//
//  return RUN_ALL_TESTS();
//}
