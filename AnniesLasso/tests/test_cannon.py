#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Unit tests for the Cannon model class and associated functions.
"""

import numpy as np
import unittest
from AnniesLasso import cannon


# Test the individual fitting functions first, then we can generate some
# real 'fake' data for the full Cannon model test.