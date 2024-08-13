from djongo import models


class Topic(models.Model):
    category = models.CharField(max_length=50)
    emotions = models.JSONField()

    class Meta:
        abstract = True


class Review(models.Model):
    text = models.TextField()
    topics = models.ArrayField(model_container=Topic)

    def __str__(self) -> str:
        return self.text
