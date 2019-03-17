# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import requests
import json

class HassApi:
    """ Home Assistant REST API Calls """

    def __init__(self, url, token):
        self.url = url
        self.token = token
        
    def TriggerAutomation(self, name):
        url = self.url + "/api/services/automation/trigger"
        headers = {
            'Authorization': "Bearer " + self.token,
        }
        payload = {"entity_id": name}
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        print(response.text)

    def TurnOnLight(self, name, color):
        url = self.url + "/api/services/light/turn_on"
        headers = {
            'Authorization': "Bearer " + self.token,
        }
        payload = {"entity_id": name, "color_name": color}
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        print(response.text)

    def PlayPlaylist(self, name, contentId):
        url = self.url + "/api/services/media_player/play_media"
        headers = {
            'Authorization': "Bearer " + self.token,
        }
        payload = {"entity_id": name, "media_content_type": "playlist", "media_content_id": contentId}
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        print(response.text)

    def PlaySong(self, name, contentId):
        url = self.url + "/api/services/media_player/play_media"
        headers = {
            'Authorization': "Bearer " + self.token,
        }
        payload = {"entity_id": name, "media_content_type": "music", "media_content_id": contentId}
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        print(response.text)

    def PauseMedia(self, name):
        url = self.url + "/api/services/media_player/media_pause"
        headers = {
            'Authorization': "Bearer " + self.token,
        }
        payload = {"entity_id": name}
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        print(response.text)

    def PlayMedia(self, name):
        url = self.url + "/api/services/media_player/media_play"
        headers = {
            'Authorization': "Bearer " + self.token,
        }
        payload = {"entity_id": name}
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        print(response.text)

    def TurnOnSwitch(self, name):
        url = self.url + "/api/services/switch/turn_on"
        headers = {
            'Authorization': "Bearer " + self.token,
        }
        payload = {"entity_id": name}
        response = requests.post(url, data=json.dumps(payload), headers=headers)
        print(response.text)

