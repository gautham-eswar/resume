import React from 'react';
import { Link as RouterLink } from 'react-router-dom';
import { Box, Flex, Link, Heading, Spacer } from '@chakra-ui/react';

function Navbar() {
  return (
    <Box bg="white" px={4} shadow="sm">
      <Flex h={16} alignItems="center" maxW="container.xl" mx="auto">
        <Heading size="md" color="blue.600">
          <Link as={RouterLink} to="/" _hover={{ textDecoration: 'none' }}>
            Resume Optimizer
          </Link>
        </Heading>
        <Spacer />
        <Flex gap={6}>
          <Link as={RouterLink} to="/" color="gray.600" fontWeight="medium">
            Optimize
          </Link>
          <Link as={RouterLink} to="/about" color="gray.600" fontWeight="medium">
            About
          </Link>
        </Flex>
      </Flex>
    </Box>
  );
}

export default Navbar; 