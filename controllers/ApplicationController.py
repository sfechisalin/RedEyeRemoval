ALLOWED_NO_ATTEMPTS = 10

class ApplicationController:
    def __init__(self, user_service, eye_fixer):
        self.__username = None
        self.__password = None
        self.__user_service = user_service
        self.__eye_fixer = eye_fixer

    def fix_image(self, image):
        if self.__username != None:
            self.__user_service.update_user_attempts(self.__username)
            return self.__eye_fixer.fix(image)
        else:
            return False

    def check_is_allowed(self, username):
        if self.__user_service.get_no_attempts(username) > ALLOWED_NO_ATTEMPTS:
            return False
        return True

    def register(self, username, passord):
        self.__username = username
        return self.__user_service.add_user(username, passord)

    def login(self, username, password):
        self.__username  = username
        return self.__user_service.exists_user(username, password)