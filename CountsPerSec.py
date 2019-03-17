# Original Code: https://github.com/nrsyed/computer-vision/blob/master/multithread/CountsPerSec.py
# Modified for use in PyPotter
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from datetime import datetime

class CountsPerSec:
    """
    Class that tracks the number of occurrences ("counts") of an
    arbitrary event and returns the frequency in occurrences
    (counts) per second. The caller must increment the count.
    """

    def __init__(self):
        self._SmoothingFactor = 90
        self._timeList = []

    def countsPerSec(self):
        self._timeList.append(datetime.now())

        if (len(self._timeList) > self._SmoothingFactor):
            self._timeList.pop(0)

        elapsed_time = (self._timeList[-1] - self._timeList[0]).total_seconds()

        if (elapsed_time > 0):
            return len(self._timeList) / elapsed_time

        return 0